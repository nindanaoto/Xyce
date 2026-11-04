"""Exercise native diode charge, capacitance, and analytic sensitivities.

Run with the built Xyce executable as the first argument. No PDK, ngspice,
third-party Python packages, or external regression checkout is required.
"""
import argparse
from bisect import bisect_left
import csv
import math
from pathlib import Path
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def charge(voltage, cap, potential, grading, fit):
    """Integrate depletion capacitance, continuing from its transition value."""
    transition = potential * fit

    def integral(v):
        # expm1/log1p keep the independent reference accurate near grading=1.
        return -cap * potential * math.expm1(
            (1 - grading) * math.log1p(-v / potential)) / (1 - grading)

    if voltage < transition:
        return integral(voltage)
    c = cap * (1 - fit) ** (-grading)
    slope = c * grading / (potential * (1 - fit))
    delta = voltage - transition
    return integral(transition) + c * delta + slope * delta * delta / 2


def run(xyce, root, name, deck, sensitivity=False):
    directory = root / name
    directory.mkdir()
    path = directory / 'test.cir'
    path.write_text(deck)
    result = subprocess.run([str(xyce), str(path)], cwd=directory,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=60)
    (directory / 'run.log').write_text(result.stdout)
    require(result.returncode == 0, f'{name}: Xyce failed:\n{result.stdout}')
    output = Path(str(path) + ('.SENS.csv' if sensitivity else '.csv'))
    with output.open() as stream:
        rows = [dict((k, float(v)) for k, v in row.items())
                for row in csv.DictReader(stream)]
    require(rows and all(math.isfinite(v) for row in rows for v in row.values()),
            f'{name}: missing or nonfinite measurements')
    return rows


def check_charge(xyce, root):
    cases = [('unequal', .25e-12, .2, .2, .4),
             ('default_fit', .25e-12, .2, .2, .5),
             ('zero_sidewall', 0, .2, .2, .4),
             ('matching', .25e-12, .729, .44, .5),
             ('near_one', .25e-12, .2, .9999, .4)]
    for level in (1, 2):
        for name, cap, phi, grading, fit in cases:
            # The narrow sweep resolves both sides of the sidewall transition;
            # the broad sweep also exercises reverse bias and bottom continuation.
            transition = phi * fit
            for sweep, start, stop, step in (
                    ('broad', -.2, .6, .001),
                    ('transition', transition - 1e-6, transition + 1e-6, 1e-6)):
                label = f'charge_l{level}_{name}_{sweep}'
                rows = run(xyce, root, label, f'''* Independent diode charge regression
.model DSW D LEVEL={level} IS=1e-14 CJ=1p M=.44 VJ=.729
+ CJSW={cap:.17g} MJSW={grading} PHP={phi} FC=.5 FCS={fit} TNOM=27
VDRV node 0 0
DTEST node 0 DSW AREA=1 PJ=1
.options device temp=27
.dc VDRV {start:.17g} {stop:.17g} {step:.17g}
.print DC FORMAT=CSV PRECISION=17 V(node) N(DTEST:qd) N(DTEST:cd)
.end
''')
                require(len(rows) == (801 if sweep == 'broad' else 3),
                        f'{label}: incomplete sweep ({len(rows)} rows)')
                for row in rows:
                    voltage = row['V(NODE)']
                    expected = (charge(voltage, 1e-12, .729, .44, .5)
                                + charge(voltage, cap, phi, grading, fit))
                    require(math.isclose(row['N(DTEST:QD)'], expected,
                                         rel_tol=1e-8, abs_tol=1e-24),
                            f'{label}: Q({voltage})={row["N(DTEST:QD)"]}, expected {expected}')
                if sweep == 'broad':
                    # Check native dQ/dV away from the transition as well, on
                    # reverse bias and on both forward charge branches.
                    for index in (100, 250, 400, 700):
                        left, middle, right = rows[index - 1:index + 2]
                        derivative = ((right['N(DTEST:QD)'] - left['N(DTEST:QD)'])
                                      / (right['V(NODE)'] - left['V(NODE)']))
                        require(math.isclose(derivative, middle['N(DTEST:CD)'], rel_tol=1e-4),
                                f'{label}: dQ/dV mismatch at {middle["V(NODE)"]}')
                # Differentiate the *native* charge across the transition, rather
                # than merely checking the derivative of the reference formula.
                if sweep == 'transition':
                    derivative = ((rows[2]['N(DTEST:QD)'] - rows[0]['N(DTEST:QD)'])
                                  / (rows[2]['V(NODE)'] - rows[0]['V(NODE)']))
                    require(math.isclose(derivative, rows[1]['N(DTEST:CD)'], rel_tol=1e-5),
                            f'{label}: native dQ/dV does not match capacitance')


def transient_deck(temperature=27, params=None, sensitivity=False, original=False):
    values = dict(CJSW=.25e-12, VJSW=.6, MJSW=.2, FCS=.4)
    values.update(params or {})
    bottom = '.729' if original else '.9'
    if original:
        values.update(VJSW=.2, FCS=.5)
    model = ' '.join(f'{key}={value:.17g}' for key, value in values.items())
    extra = '''.sens objfunc={V(node)} param=DSW:CJSW,DSW:VJSW,DSW:MJSW,DSW:FCS
.options sensitivity direct=1 adjoint=0 forceanalytic=1 outputscaled=0 outputunscaled=1
.print SENS FORMAT=CSV PRECISION=17
''' if sensitivity else ''
    return f'''* Rising and falling diode transition; no foundry model required
.model DSW D IS={'1e-14' if original else '1e-18'} N={1 if original else 2} CJ=1p M=.44 VJ={bottom}
+ {model} FC=.5 TNOM=27
VDRV drive 0 PWL(0 0 1n 0 2n {-.5 if original else -.8} 3n {-.5 if original else -.8} 4n 0 5n 0)
RDRV drive node 50
DTEST 0 node DSW AREA=1 PJ=1
.tran 1p 5n 0 1p
.options device temp={temperature}
.options nonlin reltol=1e-8 abstol=1e-14
.options timeint reltol=1e-8 abstol=1e-14
.print TRAN FORMAT=CSV PRECISION=17 V(node)
{extra}.end
'''


def check_endpoint(rows, name):
    require(math.isclose(rows[-1]['TIME'], 5e-9, rel_tol=0, abs_tol=1e-18),
            f'{name}: stopped at {rows[-1]["TIME"]}, expected 5 ns')


def interpolate(rows, time, column):
    index = bisect_left([row['TIME'] for row in rows], time)
    require(0 < index < len(rows), f'missing samples surrounding {time}')
    left, right = rows[index - 1], rows[index]
    fraction = (time - left['TIME']) / (right['TIME'] - left['TIME'])
    return left[column] + fraction * (right[column] - left[column])


def check_transients(xyce, root):
    rows = run(xyce, root, 'original_reproducer', transient_deck(original=True))
    check_endpoint(rows, 'original_reproducer')
    for temperature in (-40, 27, 125):
        name = f'transient_{temperature}'
        rows = run(xyce, root, name, transient_deck(temperature))
        check_endpoint(rows, name)
        require(min(row['V(NODE)'] for row in rows) < -.7,
                f'{name}: did not traverse the forward transition')
        require(abs(rows[-1]['V(NODE)']) < 1e-6,
                f'{name}: did not return through the transition')


def check_sensitivities(xyce, root):
    rows = run(xyce, root, 'analytic_sensitivity', transient_deck(sensitivity=True), True)
    check_endpoint(rows, 'analytic_sensitivity')
    for parameter, value in (('CJSW', .25e-12), ('VJSW', .6), ('MJSW', .2), ('FCS', .4)):
        step = value * 1e-3
        plus = run(xyce, root, f'{parameter}_plus', transient_deck(params={parameter: value + step}))
        minus = run(xyce, root, f'{parameter}_minus', transient_deck(params={parameter: value - step}))
        check_endpoint(plus, f'{parameter}_plus')
        check_endpoint(minus, f'{parameter}_minus')
        column = f'd_{{V(NODE)}}/d_DSW:{parameter}_dir'
        # Compare parameter-scaled derivatives to give a common voltage scale.
        # Samples cover both ramps and lie above the sidewall transition.
        for time in (1.375e-9, 1.4e-9, 1.6e-9, 1.8e-9,
                     3.2e-9, 3.4e-9, 3.7e-9, 3.75e-9):
            analytic = interpolate(rows, time, column) * value
            numerical = (interpolate(plus, time, 'V(NODE)')
                         - interpolate(minus, time, 'V(NODE)')) * value / (2 * step)
            require(math.isclose(analytic, numerical, rel_tol=5e-3, abs_tol=2e-7),
                    f'{parameter} at {time}: analytic={analytic}, finite difference={numerical}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xyce', type=Path)
    parser.add_argument('--output', type=Path, help='retain decks, logs and results in a new directory')
    args = parser.parse_args()
    xyce = args.xyce.resolve(strict=True)

    def check(root):
        check_charge(xyce, root)
        print('PASS: native charge integral and dQ/dV, levels 1 and 2', flush=True)
        check_transients(xyce, root)
        print('PASS: original reproducer and cold/nominal/hot transient endpoints', flush=True)
        check_sensitivities(xyce, root)
        print('PASS: analytic sensitivities versus central finite differences', flush=True)

    if args.output:
        args.output.mkdir(parents=True)
        check(args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='xyce-diode-') as directory:
            check(Path(directory))


if __name__ == '__main__':
    main()
