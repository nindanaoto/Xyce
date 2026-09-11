// SPDX-License-Identifier: GPL-3.0-or-later
// Check local CSR ordering, distributed import, refactor, factor reuse,
// transpose solves, and multiple right-hand sides against known solutions.
#include <Xyce_config.h>
#ifdef Xyce_PARALLEL_MPI
#include <mpi.h>
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif
#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <N_LAS_EpetraProblem.h>
#include <N_LAS_KLSSolver.h>
#include <N_UTL_OptionBlock.h>
#include <algorithm>
#include <cmath>
#include <iostream>

int runCase(const Epetra_Comm & comm, bool permutedColumns, int rhsCount,
            const char * backend)
{
  const int n = 4;
  Epetra_Map rows(n, 0, comm);
  int columnIDs[n] = {0, 1, 2, 3};
  if (permutedColumns)
    std::reverse(columnIDs, columnIDs + n);
  Epetra_Map columns(-1, n, columnIDs, 0, comm);
  Epetra_CrsMatrix matrix(Copy, rows, columns, 2);
  int errors = 0;
  for (int row = 0; row < rows.NumMyElements(); ++row)
  {
    int gid = rows.GID(row);
    int cols[2] = {gid, (gid + 1) % n};
    double vals[2] = {4.0 + gid, 0.5 + 0.25 * gid};
    errors += matrix.InsertGlobalValues(gid, 2, vals, cols) < 0;
  }
  errors += matrix.FillComplete(rows, rows) != 0;
  errors += matrix.OptimizeStorage() != 0;
  Epetra_MultiVector solution(rows, rhsCount), rhs(rows, rhsCount), exact(rows, rhsCount);
  Teuchos::RCP<Epetra_LinearProblem> epetra = Teuchos::rcp(
    new Epetra_LinearProblem(&matrix, &solution, &rhs));
  Xyce::Linear::EpetraProblem problem(epetra);
  Xyce::Util::OptionBlock options;
  options.addParam(Xyce::Util::Param("KLS_BACKEND", backend));
  options.addParam(Xyce::Util::Param("KLS_THREADS", 1));
  Xyce::Linear::KLSSolver solver(problem, options);
  for (int step = 0; step < 4; ++step)
  {
    if (step == 1)
      for (int row = 0; row < rows.NumMyElements(); ++row)
      {
        int gid = rows.GID(row);
        double increment = 0.75;
        errors += matrix.SumIntoGlobalValues(gid, 1, &increment, &gid) != 0;
      }
    for (int col = 0; col < rhsCount; ++col)
      for (int row = 0; row < rows.NumMyElements(); ++row)
        exact[col][row] = (rows.GID(row) + 1.0) * (col + 1.0) + step * 0.25;
    const bool transpose = step == 3;
    errors += matrix.Multiply(transpose, exact, rhs) != 0;
    solution.PutScalar(0.0);
    errors += solver.doSolve(step >= 2, transpose) != 0;
    double localError = 0.0, globalError = 0.0;
    for (int col = 0; col < rhsCount; ++col)
      for (int row = 0; row < rows.NumMyElements(); ++row)
      {
        const double value = solution[col][row];
        if (!std::isfinite(value))
          localError = 1.0;
        else
          localError = std::max(localError, std::abs(value - exact[col][row]));
      }
    comm.MaxAll(&localError, &globalError, 1);
    if (globalError > 1e-10)
    {
      ++errors;
      if (comm.MyPID() == 0)
        std::cerr << backend << " permuted=" << permutedColumns
          << " rhs=" << rhsCount << " step=" << step
          << " error=" << globalError << '\n';
    }
  }
  return errors;
}

int main(int argc, char ** argv)
{
#ifdef Xyce_PARALLEL_MPI
  MPI_Init(&argc, &argv);
#endif
  int result = 0;
  {
#ifdef Xyce_PARALLEL_MPI
    Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
    Epetra_SerialComm comm;
#endif
    int errors = 0;
    for (const char * backend : {"AUTO", "SERIAL"})
      for (bool permuted : {false, true})
        for (int rhsCount : {1, 2})
          errors += runCase(comm, permuted, rhsCount, backend);
    comm.MaxAll(&errors, &result, 1);
    if (comm.MyPID() == 0)
      std::cout << (result ? "KLS adapter regression FAILED" : "KLS adapter regression passed") << std::endl;
  }
#ifdef Xyce_PARALLEL_MPI
  MPI_Finalize();
#endif
  return result ? 1 : 0;
}
