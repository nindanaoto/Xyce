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

#include <Xyce_config.h>

#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <Epetra_Comm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Import.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>

#include <N_LAS_ThirdPartySolver.h>

#include <N_ERH_ErrorMgr.h>
#include <N_LAS_EpetraHelpers.h>
#include <N_LAS_EpetraProblem.h>
#include <N_LAS_MultiVector.h>
#include <N_LAS_Problem.h>
#include <N_UTL_ExtendedString.h>
#include <N_UTL_FeatureTest.h>
#include <N_UTL_OptionBlock.h>
#include <N_UTL_Timer.h>

#ifdef Xyce_CKTSO
#include <cktso.h>
#endif
#ifdef Xyce_SUBTREELU
#include <subtree_lu.h>
#endif

namespace {
enum { TP_OK = 0, TP_INVALID = -1001, TP_UNSUPPORTED = -1002 };
}

namespace Xyce {
namespace Linear {

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::ThirdPartySolver
// Purpose       :
// Special Notes :
// Scope         : Public
//-----------------------------------------------------------------------------
ThirdPartySolver::ThirdPartySolver(
  Backend               backend,
  Problem &             problem,
  Util::OptionBlock &   options)
  : Solver(problem, false),
    problem_(0),
    backend_(backend),
    solver_(0),
    threads_(1),
    pivotTolerance_(0.001),
    memoryGrowth_(1.5),
    ordering_(0),
    scaling_(0),
    automaticThreads_(false),
    fastFactor_(true),
    analyzed_(false),
    factored_(false),
    outputLS_(0),
    outputBaseLS_(0),
    outputFailedLS_(0),
    options_(new Util::OptionBlock(options)),
    timer_(Teuchos::rcp(new Util::Timer()))
{
  EpetraProblem & eprob = dynamic_cast<EpetraProblem &>(lasProblem_);
  problem_ = &(eprob.epetraObj());

  setOptions(options);
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::~ThirdPartySolver
// Purpose       :
// Special Notes :
// Scope         : Public
//-----------------------------------------------------------------------------
ThirdPartySolver::~ThirdPartySolver()
{
  clearAnalysis_();
  delete options_;
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::setOptions
// Purpose       :
// Special Notes :
// Scope         : Public
//-----------------------------------------------------------------------------
bool ThirdPartySolver::setOptions(const Util::OptionBlock & OB)
{
  threads_ = 1;
  pivotTolerance_ = 0.001;
  memoryGrowth_ = 1.5;
  ordering_ = 0;
  scaling_ = 0;
  automaticThreads_ = false;
  fastFactor_ = true;

  for (Util::ParamList::const_iterator it = OB.begin(); it != OB.end(); ++it)
  {
    const std::string tag = it->uTag();

    if (tag == "OUTPUT_LS")
      outputLS_ = it->getImmutableValue<int>();
    else if (tag == "OUTPUT_BASE_LS")
      outputBaseLS_ = it->getImmutableValue<int>();
    else if (tag == "OUTPUT_FAILED_LS")
      outputFailedLS_ = it->getImmutableValue<int>();
    else if ((backend_ == CKTSO_BACKEND && tag == "CKTSO_THREADS") ||
             (backend_ == SUBTREELU_BACKEND && tag == "SUBTREELU_THREADS"))
      threads_ = it->getImmutableValue<int>();
    else if ((backend_ == CKTSO_BACKEND && tag == "CKTSO_PIVOT_TOLERANCE") ||
             (backend_ == SUBTREELU_BACKEND && tag == "SUBTREELU_PIVOT_TOLERANCE"))
      pivotTolerance_ = it->getImmutableValue<double>();
    else if (backend_ == CKTSO_BACKEND && tag == "CKTSO_ORDERING")
      ordering_ = it->getImmutableValue<int>();
    else if (backend_ == CKTSO_BACKEND && tag == "CKTSO_SCALE")
      scaling_ = it->getImmutableValue<int>();
    else if (backend_ == CKTSO_BACKEND && tag == "CKTSO_AUTO_THREADS")
      automaticThreads_ = it->getImmutableValue<int>() != 0;
    else if (backend_ == CKTSO_BACKEND && tag == "CKTSO_FAST_FACTOR")
      fastFactor_ = it->getImmutableValue<int>() != 0;
    else if (backend_ == SUBTREELU_BACKEND && tag == "SUBTREELU_MEMORY_GROWTH")
      memoryGrowth_ = it->getImmutableValue<double>();
  }

  delete options_;
  options_ = new Util::OptionBlock(OB);
  clearAnalysis_();

  return true;
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::doSolve
// Purpose       :
// Special Notes :
// Scope         : Public
//-----------------------------------------------------------------------------
int ThirdPartySolver::doSolve(bool reuse_factors, bool transpose)
{
  timer_->resetStartTime();

  int linearStatus = 0;
  int backendStatus = TP_OK;
  Epetra_LinearProblem * prob = problem_;

  static int failure_number = 0;
  static int file_number = 1;
  static int base_file_number = 1;

  if (outputLS_)
  {
    if (!(file_number % outputLS_) && !reuse_factors)
      Xyce::Linear::writeToFile(*prob, "Transformed", file_number,
                                (file_number == 1));
  }
  if (outputBaseLS_)
  {
    if (!(base_file_number % outputBaseLS_) && !reuse_factors)
      Xyce::Linear::writeToFile(*problem_, "Base", base_file_number,
                                (base_file_number == 1));
  }

  Epetra_CrsMatrix * inputMatrix = dynamic_cast<Epetra_CrsMatrix *>(prob->GetMatrix());
  if (!inputMatrix)
    return -1;

  if (DEBUG_LINEAR)
    inputMatrix->SetTracebackMode(2);
  else
    inputMatrix->SetTracebackMode(0);

  prob = importToSerial_();
  Epetra_CrsMatrix * matrix = dynamic_cast<Epetra_CrsMatrix *>(prob->GetMatrix());
  if (!matrix)
    return -1;

  const int myPID = matrix->Comm().MyPID();

  if (myPID == 0)
  {
    if (!analyzed_)
      backendStatus = analyze_(prob);

    if (backendStatus == TP_OK && (!reuse_factors || !factored_))
    {
      backendStatus = factor_(prob);
    }

    if (backendStatus == TP_OK)
      backendStatus = solve_(prob, transpose);
  }

  int localFailure = (backendStatus == TP_OK) ? 0 : 1;
  int globalFailure = 0;
  matrix->Comm().MaxAll(&localFailure, &globalFailure, 1);

  if (globalFailure)
  {
    linearStatus = -1;
    prob->GetLHS()->PutScalar(0.0);
    if (myPID == 0)
    {
      Report::UserWarning0()
        << (backend_ == CKTSO_BACKEND ? "CKTSO" : "SubtreeLU")
        << " linear solve failed (" << backendStatus
        << "), returning zero solution to nonlinear solver!";
    }
    if (outputFailedLS_)
    {
      failure_number++;
      Xyce::Linear::writeToFile(*prob, "Failed", failure_number,
                                (failure_number == 1));
    }
  }

  exportToGlobal_();

  if (VERBOSE_LINEAR)
  {
    const double endSolveTime = timer_->elapsedTime();
    Xyce::lout() << "  " << (backend_ == CKTSO_BACKEND ? "CKTSO" : "SubtreeLU")
                 << " Solve Time: " << endSolveTime << std::endl;
  }

  if (DEBUG_LINEAR && linearStatus == 0)
  {
    const int numrhs = prob->GetLHS()->NumVectors();
    std::vector<double> resNorm(numrhs, 0.0), bNorm(numrhs, 0.0);
    Epetra_MultiVector res(prob->GetLHS()->Map(), numrhs);
    const bool oldTrans = prob->GetOperator()->UseTranspose();
    prob->GetOperator()->SetUseTranspose(transpose);
    prob->GetOperator()->Apply(*(prob->GetLHS()), res);
    prob->GetOperator()->SetUseTranspose(oldTrans);
    res.Update(1.0, *(prob->GetRHS()), -1.0);
    res.Norm2(&resNorm[0]);
    prob->GetRHS()->Norm2(&bNorm[0]);
    Xyce::lout() << "Linear System Residual ("
                 << (backend_ == CKTSO_BACKEND ? "CKTSO" : "SubtreeLU")
                 << "): " << std::endl;
    for (int i = 0; i < numrhs; ++i)
    {
      if (bNorm[i] > 0.0)
        std::cout << "  Problem " << i << " : " << (resNorm[i] / bNorm[i]) << std::endl;
      else
        std::cout << "  Problem " << i << " : " << resNorm[i] << std::endl;
    }
  }

  if (outputLS_)
  {
    if (!(file_number % outputLS_))
    {
      Teuchos::RCP<Problem> las_prob =
        Teuchos::rcp(new EpetraProblem(Teuchos::rcp(prob, false)));
      std::stringstream file_name("Transformed_Soln");
      file_name << file_number << ".mm";
      las_prob->getLHS()->writeToFile(file_name.str().c_str(), false, true);
    }
    file_number++;
  }
  if (outputBaseLS_)
  {
    if (!(base_file_number % outputBaseLS_))
    {
      std::stringstream file_name("Base_Soln");
      file_name << base_file_number << ".mm";
      lasProblem_.getLHS()->writeToFile(file_name.str().c_str(), false, true);
    }
    base_file_number++;
  }

  solutionTime_ = timer_->elapsedTime();

  if (VERBOSE_LINEAR)
    Xyce::lout() << "Total Linear Solution Time ("
                 << (backend_ == CKTSO_BACKEND ? "CKTSO" : "SubtreeLU")
                 << "): " << solutionTime_ << std::endl;

  return linearStatus;
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::analyze_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int ThirdPartySolver::analyze_(Epetra_LinearProblem * problem)
{
  Epetra_CrsMatrix * matrix = dynamic_cast<Epetra_CrsMatrix *>(problem->GetMatrix());
  if (!matrix)
    return TP_INVALID;

  const int buildStatus = buildCSR_(matrix);
  if (buildStatus != TP_OK)
    return buildStatus;

  if (threads_ < 1 || pivotTolerance_ < 0.0 || pivotTolerance_ > 1.0 ||
      memoryGrowth_ <= 0.0)
    return TP_INVALID;

  if (!solver_ && backend_ == CKTSO_BACKEND)
  {
#ifdef Xyce_CKTSO
    ICktSo instance = 0;
    int * iparm = 0;
    const long long * oparm = 0;
    const int createStatus = CKTSO_CreateSolver(&instance, &iparm, &oparm);
    if (createStatus != 0)
      return createStatus;
    iparm[0] = 1;
    iparm[1] = static_cast<int>(std::lround(pivotTolerance_ * 1.0e6));
    iparm[2] = ordering_;
    iparm[7] = scaling_;
    iparm[9] = automaticThreads_ ? 1 : 0;
    solver_ = instance;
#else
    return TP_UNSUPPORTED;
#endif
  }
  else if (!solver_)
  {
#ifdef Xyce_SUBTREELU
    subtree_lu::SubtreeLU<int, double> * instance =
      new subtree_lu::SubtreeLU<int, double>();
    instance->parm[subtree_lu::I_PIVTOL] =
      static_cast<long long>(std::lround(pivotTolerance_ * 1.0e6));
    instance->parm[subtree_lu::I_MEM_FACTOR] =
      static_cast<long long>(std::lround(memoryGrowth_ * 100.0));
    solver_ = instance;
#else
    return TP_UNSUPPORTED;
#endif
  }

  const int n = matrix->NumMyRows();
  int status = TP_UNSUPPORTED;
  if (backend_ == CKTSO_BACKEND)
  {
#ifdef Xyce_CKTSO
    status = CKTSO_Analyze(static_cast<ICktSo>(solver_), false, n,
                           &rowPtr_[0], colIdx_.empty() ? 0 : &colIdx_[0],
                           values_.empty() ? 0 : &values_[0], threads_);
#endif
  }
  else
  {
#ifdef Xyce_SUBTREELU
    status = static_cast<subtree_lu::SubtreeLU<int, double> *>(solver_)->analyze(
      n, &rowPtr_[0], colIdx_.empty() ? 0 : &colIdx_[0],
      values_.empty() ? 0 : &values_[0], threads_, 0, 0);
#endif
  }
  analyzed_ = (status == TP_OK);
  factored_ = false;
  return status;
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::factor_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int ThirdPartySolver::factor_(Epetra_LinearProblem * problem)
{
  Epetra_CrsMatrix * matrix = dynamic_cast<Epetra_CrsMatrix *>(problem->GetMatrix());
  if (!matrix || !solver_)
    return TP_INVALID;

  int status = updateValues_(matrix);
  if (status != TP_OK)
    return status;

  if (factored_)
  {
    if (backend_ == CKTSO_BACKEND)
    {
#ifdef Xyce_CKTSO
      status = CKTSO_Refactorize(static_cast<ICktSo>(solver_), &values_[0]);
      if (status != TP_OK)
        status = CKTSO_Factorize(static_cast<ICktSo>(solver_), &values_[0], fastFactor_);
#endif
    }
    else
    {
#ifdef Xyce_SUBTREELU
      subtree_lu::SubtreeLU<int, double> * instance =
        static_cast<subtree_lu::SubtreeLU<int, double> *>(solver_);
      status = instance->refactorize(&values_[0]);
      if (status != TP_OK)
        status = instance->factorize(&values_[0]);
#endif
    }
  }
  else
  {
    if (backend_ == CKTSO_BACKEND)
    {
#ifdef Xyce_CKTSO
      status = CKTSO_Factorize(static_cast<ICktSo>(solver_), &values_[0], fastFactor_);
#endif
    }
    else
    {
#ifdef Xyce_SUBTREELU
      status = static_cast<subtree_lu::SubtreeLU<int, double> *>(solver_)->factorize(&values_[0]);
#endif
    }
  }

  factored_ = (status == TP_OK);
  return status;
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::solve_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int ThirdPartySolver::solve_(Epetra_LinearProblem * problem, bool transpose)
{
  if (!solver_ || !factored_)
    return TP_INVALID;

  if (transpose && backend_ == SUBTREELU_BACKEND)
    return TP_UNSUPPORTED;

  Epetra_MultiVector * X = problem->GetLHS();
  Epetra_MultiVector * B = problem->GetRHS();
  if (!X || !B || X->NumVectors() != B->NumVectors())
    return TP_INVALID;

  double ** rhsPtrs = 0;
  double ** lhsPtrs = 0;
  if (B->ExtractView(&rhsPtrs) != 0 || X->ExtractView(&lhsPtrs) != 0)
    return TP_INVALID;

  int status = TP_OK;
  for (int rhs = 0; rhs < B->NumVectors(); ++rhs)
  {
    if (backend_ == CKTSO_BACKEND)
    {
#ifdef Xyce_CKTSO
      status = CKTSO_Solve(static_cast<ICktSo>(solver_), rhsPtrs[rhs],
                           lhsPtrs[rhs], false, transpose);
#endif
    }
    else
    {
#ifdef Xyce_SUBTREELU
      status = static_cast<subtree_lu::SubtreeLU<int, double> *>(solver_)->solve(
        rhsPtrs[rhs], lhsPtrs[rhs]);
#endif
    }

    if (status != TP_OK)
      return status;
  }

  return status;
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::buildCSR_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int ThirdPartySolver::buildCSR_(Epetra_CrsMatrix * matrix)
{
  if (!matrix)
    return TP_INVALID;

  if (matrix->NumGlobalRows() != matrix->NumGlobalCols() ||
      matrix->NumMyRows() != matrix->NumGlobalRows())
    return TP_UNSUPPORTED;

  if (matrix->NumMyRows() < 0 || matrix->NumMyNonzeros() < 0)
    return TP_INVALID;

  const int n = matrix->NumMyRows();
  const int nnz = matrix->NumMyNonzeros();

  rowPtr_.assign(static_cast<std::size_t>(n) + 1, 0);
  colIdx_.assign(static_cast<std::size_t>(nnz), 0);
  values_.assign(static_cast<std::size_t>(nnz), 0.0);

  int offset = 0;
  for (int row = 0; row < n; ++row)
  {
    int numEntries = 0;
    int * indices = 0;
    double * rowValues = 0;
    const int ierr = matrix->ExtractMyRowView(row, numEntries, rowValues, indices);
    if (ierr != 0)
      return TP_INVALID;

    rowPtr_[static_cast<std::size_t>(row)] = offset;
    for (int entry = 0; entry < numEntries; ++entry)
    {
      if (indices[entry] < 0 || indices[entry] >= n)
        return TP_UNSUPPORTED;
      colIdx_[static_cast<std::size_t>(offset)] = indices[entry];
      values_[static_cast<std::size_t>(offset)] = rowValues[entry];
      ++offset;
    }
  }

  if (offset != nnz)
    return TP_INVALID;

  rowPtr_[static_cast<std::size_t>(n)] = offset;
  return TP_OK;
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::updateValues_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int ThirdPartySolver::updateValues_(Epetra_CrsMatrix * matrix)
{
  if (!matrix)
    return TP_INVALID;

  const int n = matrix->NumMyRows();
  if (static_cast<std::size_t>(matrix->NumMyNonzeros()) != values_.size() ||
      rowPtr_.size() != static_cast<std::size_t>(n) + 1)
  {
    clearAnalysis_();
    return TP_UNSUPPORTED;
  }

  int offset = 0;
  for (int row = 0; row < n; ++row)
  {
    int numEntries = 0;
    double * rowValues = 0;
    const int ierr = matrix->ExtractMyRowView(row, numEntries, rowValues);
    if (ierr != 0)
      return TP_INVALID;

    if (rowPtr_[static_cast<std::size_t>(row)] != offset ||
        rowPtr_[static_cast<std::size_t>(row) + 1] != offset + numEntries)
    {
      clearAnalysis_();
      return TP_UNSUPPORTED;
    }

    for (int entry = 0; entry < numEntries; ++entry)
      values_[static_cast<std::size_t>(offset++)] = rowValues[entry];
  }

  return TP_OK;
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::clearAnalysis_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
void ThirdPartySolver::clearAnalysis_()
{
  if (solver_)
  {
    if (backend_ == CKTSO_BACKEND)
    {
#ifdef Xyce_CKTSO
      CKTSO_DestroySolver(static_cast<ICktSo>(solver_));
#endif
    }
    else
    {
#ifdef Xyce_SUBTREELU
      delete static_cast<subtree_lu::SubtreeLU<int, double> *>(solver_);
#endif
    }
    solver_ = 0;
  }
  analyzed_ = false;
  factored_ = false;
  rowPtr_.clear();
  colIdx_.clear();
  values_.clear();
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::importToSerial_
// Purpose       : Import a distributed matrix for a serial direct solver.
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
Epetra_LinearProblem * ThirdPartySolver::importToSerial_()
{
#ifdef Xyce_PARALLEL_MPI
  Epetra_CrsMatrix * origMat = dynamic_cast<Epetra_CrsMatrix *>(problem_->GetOperator());

  if (serialMap_ == Teuchos::null)
  {
    const Epetra_Map & origMap = origMat->RowMap();
    const int myPID = origMap.Comm().MyPID();
    const int numGlobalElements = origMap.NumGlobalElements();
    int numMyElements = numGlobalElements;
    if (myPID != 0)
      numMyElements = 0;
    serialMap_ = Teuchos::rcp(new Epetra_Map(-1, numMyElements, 0, origMap.Comm()));
    const int numVectors = problem_->GetRHS()->NumVectors();

    serialLHS_ = Teuchos::rcp(new Epetra_MultiVector(*serialMap_, numVectors));
    serialRHS_ = Teuchos::rcp(new Epetra_MultiVector(*serialMap_, numVectors));
    serialImporter_ = Teuchos::rcp(new Epetra_Import(*serialMap_, origMap));
    serialMat_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *serialMap_, 0));
    serialProblem_ =
      Teuchos::rcp(new Epetra_LinearProblem(&*serialMat_, &*serialLHS_, &*serialRHS_));
  }

  serialMat_->Import(*origMat, *serialImporter_, Insert);
  serialMat_->FillComplete();
  serialMat_->OptimizeStorage();
  serialLHS_->Import(*problem_->GetLHS(), *serialImporter_, Insert);
  serialRHS_->Import(*problem_->GetRHS(), *serialImporter_, Insert);

  return &*serialProblem_;
#else
  return problem_;
#endif
}

//-----------------------------------------------------------------------------
// Function      : ThirdPartySolver::exportToGlobal_
// Purpose       : Export the serial solution to the global vector.
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int ThirdPartySolver::exportToGlobal_()
{
#ifdef Xyce_PARALLEL_MPI
  problem_->GetLHS()->Export(*serialLHS_, *serialImporter_, Insert);
#endif
  return 0;
}

} // namespace Linear
} // namespace Xyce
