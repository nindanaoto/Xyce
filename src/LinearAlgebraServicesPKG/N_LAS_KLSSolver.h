//-------------------------------------------------------------------------
//   Copyright 2002-2026 National Technology & Engineering Solutions of
//   Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
//   NTESS, the U.S. Government retains certain rights in this software.
//
//   This file is part of the Xyce(TM) Parallel Electrical Simulator.
//
//   Xyce(TM) is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   Xyce(TM) is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with Xyce(TM).
//   If not, see <http://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// Purpose        : KLS direct linear solver interface
//
// Special Notes  :
//
//-----------------------------------------------------------------------------

#ifndef Xyce_N_LAS_KLSSolver_h
#define Xyce_N_LAS_KLSSolver_h

#include <vector>

#include <N_LAS_fwd.h>
#include <N_UTL_fwd.h>

#include <N_LAS_Solver.h>
#include <Teuchos_RCP.hpp>

#include <kls/kls.h>

class Epetra_CrsMatrix;
class Epetra_Import;
class Epetra_LinearProblem;
class Epetra_Map;
class Epetra_MultiVector;

namespace Xyce {
namespace Linear {

//-----------------------------------------------------------------------------
// Class         : KLSSolver
// Purpose       : Wraps the standalone KLS sparse direct solver for Xyce.
//-----------------------------------------------------------------------------
class KLSSolver : public Solver
{
public:
  KLSSolver(Problem & problem, Util::OptionBlock & options);
  ~KLSSolver();

  bool setOptions(const Util::OptionBlock & OB);

  int doSolve(bool reuse_factors, bool transpose = false);

private:
  int analyze_(Epetra_LinearProblem * problem);
  int factor_(Epetra_LinearProblem * problem);
  int refactorSolve_(Epetra_LinearProblem * problem);
  int solve_(Epetra_LinearProblem * problem, bool transpose);
  int buildCSR_(Epetra_CrsMatrix * matrix);
  int updateValues_(Epetra_CrsMatrix * matrix, bool * changed = 0);
  void clearAnalysis_();

  Epetra_LinearProblem * importToSerial_();
  int exportToGlobal_();

  Epetra_LinearProblem * problem_;

  kls_solver * solver_;
  kls_options klsOptions_;
  bool analyzed_;
  bool factored_;

  std::vector<int> rowPtr_;
  std::vector<int> colIdx_;
  std::vector<double> values_;

  int outputLS_;
  int outputBaseLS_;
  int outputFailedLS_;

  Teuchos::RCP<Epetra_Map> serialMap_;
  Teuchos::RCP<Epetra_LinearProblem> serialProblem_;
  Teuchos::RCP<Epetra_CrsMatrix> serialMat_;
  Teuchos::RCP<Epetra_MultiVector> serialLHS_;
  Teuchos::RCP<Epetra_MultiVector> serialRHS_;
  Teuchos::RCP<Epetra_Import> serialImporter_;

  Util::OptionBlock * options_;
  Teuchos::RCP<Util::Timer> timer_;
};

} // namespace Linear
} // namespace Xyce

#endif // Xyce_N_LAS_KLSSolver_h
