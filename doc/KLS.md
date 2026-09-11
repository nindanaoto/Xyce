# KLS linear solver

Enable the optional solver at configure time with `-DXyce_KLS=ON` and
`-DXyce_KLS_ROOT=/path/to/KLS`. The build also discovers a sibling `KLS`
source directory. Existing Trilinos/Xyce build requirements still apply.

Select KLS in a netlist:

```spice
.options linsol type=KLS KLS_THREADS=1
```

The MPI-enabled adapter uses the original matrix directly when there is one
MPI rank and the matrix row, column, domain, range, and vector maps agree.
Other layouts retain the import-to-rank-zero path. The shortcut changes data
handling, not the default KLS numerical policy.

## Backend selection

`KLS_BACKEND` accepts `AUTO` (default), `KLS`, or `SERIAL`, corresponding to
the KLS API backend values. For the lean serial numerical policy:

```spice
.options linsol type=KLS KLS_BACKEND=SERIAL KLS_THREADS=1
```

`SERIAL` requires exactly one KLS thread. It can still be used with multiple
MPI ranks: Xyce distributes device evaluation and KLS solves on rank zero.
Setting `KLS_THREADS=1` alone does not select this backend. SERIAL supports
AUTO/AMD, COLAMD, or NATURAL ordering; other numerical options are subject
to the KLS API's backend constraints.

## Adapter profiling

Add `KLS_PROFILE=1` to the LINSOL options to report aggregate timings when
the solver is destroyed. Profiling is off by default and reads no profiling
clock when disabled. The output reports the complete adapter time, import,
export, analysis/CSR construction, value checking/copying, factorization,
combined refactor/solve, and standalone solve time. The phase times are
exclusive; the total additionally includes adapter bookkeeping and error
handling. Counts include all adapter calls and calls using the original
matrix directly. With multiple MPI ranks, the report contains rank-zero
wall times, not a sum or maximum across ranks.

Use profiling for diagnosis and leave it off for final performance timings.

## Regression test

With `BUILD_TESTING=ON`, build `testKLSAdapter` and run
`ctest -R '^testKLSAdapter' --output-on-failure`. The tests compare against
known solutions for normal and permuted column maps, changing numeric
values, factor reuse, transpose solves, and multiple right-hand sides,
using both AUTO and SERIAL. MPI builds also register a four-rank test.
