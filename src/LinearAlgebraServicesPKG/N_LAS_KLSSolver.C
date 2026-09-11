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

#include <cstring>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Epetra_Comm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Import.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>

#include <N_LAS_KLSSolver.h>

#include <N_ERH_ErrorMgr.h>
#include <N_LAS_EpetraHelpers.h>
#include <N_LAS_EpetraProblem.h>
#include <N_LAS_MultiVector.h>
#include <N_LAS_Problem.h>
#include <N_UTL_ExtendedString.h>
#include <N_UTL_FeatureTest.h>
#include <N_UTL_OptionBlock.h>
#include <N_UTL_Timer.h>

namespace {

// No clock reads when profiling is disabled. Each accumulator is exclusive
// except total, which measures the complete adapter call.
class ProfileTimer
{
public:
  explicit ProfileTimer(double * seconds) : seconds_(seconds)
  {
    if (seconds_)
      start_ = std::chrono::steady_clock::now();
  }
  ~ProfileTimer()
  {
    if (seconds_)
      *seconds_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_).count();
  }
private:
  double * seconds_;
  std::chrono::steady_clock::time_point start_;
};

std::string upperValue(const Xyce::Util::Param & param)
{
  Xyce::ExtendedString value = param.usVal();
  value.toUpper();
  return value;
}

kls_backend parseBackend(const Xyce::Util::Param & param)
{
  const std::string value = upperValue(param);
  if (value == "AUTO") return KLS_BACKEND_AUTO;
  if (value == "KLS") return KLS_BACKEND_KLS;
  if (value == "SERIAL") return KLS_BACKEND_SERIAL;
  Xyce::Report::UserError0() << "KLS_BACKEND must be AUTO, KLS, or SERIAL";
  return KLS_BACKEND_AUTO;
}

kls_ordering parseOrdering(const Xyce::Util::Param & param,
                           kls_ordering current)
{
  const std::string value = upperValue(param);
  if (value == "AUTO")
    return KLS_ORDERING_AUTO;
  if (value == "AMD")
    return KLS_ORDERING_AMD;
  if (value == "COLAMD")
    return KLS_ORDERING_COLAMD;
  if (value == "NATURAL")
    return KLS_ORDERING_NATURAL;
  if (value == "METIS")
    return KLS_ORDERING_METIS;
  if (value == "SCOTCH")
    return KLS_ORDERING_SCOTCH;
  if (value == "AMF")
    return KLS_ORDERING_AMF;
  return current;
}

kls_orientation parseOrientation(const Xyce::Util::Param & param,
                                 kls_orientation current)
{
  const std::string value = upperValue(param);
  if (value == "AUTO")
    return KLS_ORIENTATION_AUTO;
  if (value == "NORMAL")
    return KLS_ORIENTATION_NORMAL;
  if (value == "TRANSPOSE")
    return KLS_ORIENTATION_TRANSPOSE;
  return current;
}

} // namespace

namespace Xyce {
namespace Linear {

//-----------------------------------------------------------------------------
// Function      : KLSSolver::KLSSolver
// Purpose       :
// Special Notes :
// Scope         : Public
//-----------------------------------------------------------------------------
KLSSolver::KLSSolver(
  Problem &             problem,
  Util::OptionBlock &   options)
  : Solver(problem, false),
    problem_(0),
    solver_(0),
    analyzed_(false),
    factored_(false),
    usingOriginalProblem_(false),
    profile_(false),
    outputLS_(0),
    outputBaseLS_(0),
    outputFailedLS_(0),
    options_(new Util::OptionBlock(options)),
    timer_(Teuchos::rcp(new Util::Timer()))
{
  EpetraProblem & eprob = dynamic_cast<EpetraProblem &>(lasProblem_);
  problem_ = &(eprob.epetraObj());

  kls_default_options(&klsOptions_);
  klsOptions_.record_tiny_solve_timing = 0;
  setOptions(options);
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::~KLSSolver
// Purpose       :
// Special Notes :
// Scope         : Public
//-----------------------------------------------------------------------------
KLSSolver::~KLSSolver()
{
  if (profile_ && profileStats_.calls && problem_->GetRHS()->Comm().MyPID() == 0)
  {
    Xyce::lout() << "KLS profile calls = " << profileStats_.calls << '\n'
      << "KLS profile direct calls = " << profileStats_.directCalls << '\n'
      << "KLS profile total seconds = " << profileStats_.total << '\n'
      << "KLS profile import seconds = " << profileStats_.imports << '\n'
      << "KLS profile export seconds = " << profileStats_.exports << '\n'
      << "KLS profile analysis seconds = " << profileStats_.analysis << '\n'
      << "KLS profile values seconds = " << profileStats_.values << '\n'
      << "KLS profile factor seconds = " << profileStats_.factor << '\n'
      << "KLS profile refactor solve seconds = " << profileStats_.refactorSolve << '\n'
      << "KLS profile solve seconds = " << profileStats_.solve << std::endl;
  }
  if (solver_)
    kls_destroy(solver_);
  delete options_;
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::setOptions
// Purpose       :
// Special Notes :
// Scope         : Public
//-----------------------------------------------------------------------------
bool KLSSolver::setOptions(const Util::OptionBlock & OB)
{
  kls_default_options(&klsOptions_);
  klsOptions_.record_tiny_solve_timing = 0;

  profile_ = false;
  profileStats_ = Profile();

  for (Util::ParamList::const_iterator it = OB.begin(); it != OB.end(); ++it)
  {
    const std::string tag = it->uTag();

    if (tag == "OUTPUT_LS")
      outputLS_ = it->getImmutableValue<int>();
    else if (tag == "OUTPUT_BASE_LS")
      outputBaseLS_ = it->getImmutableValue<int>();
    else if (tag == "OUTPUT_FAILED_LS")
      outputFailedLS_ = it->getImmutableValue<int>();
    else if (tag == "KLS_THREADS")
      klsOptions_.threads = it->getImmutableValue<int>();
    else if (tag == "KLS_BACKEND")
      klsOptions_.backend = parseBackend(*it);
    else if (tag == "KLS_PROFILE")
      profile_ = it->getImmutableValue<int>() != 0;
    else if (tag == "KLS_ORDERING")
      klsOptions_.ordering = parseOrdering(*it, klsOptions_.ordering);
    else if (tag == "KLS_ORIENTATION")
      klsOptions_.orientation = parseOrientation(*it, klsOptions_.orientation);
    else if (tag == "KLS_USE_BTF")
      klsOptions_.use_btf = it->getImmutableValue<int>();
    else if (tag == "KLS_SCALE")
      klsOptions_.scale = it->getImmutableValue<int>();
    else if (tag == "KLS_PIVOT_TOLERANCE")
      klsOptions_.pivot_tolerance = it->getImmutableValue<double>();
    else if (tag == "KLS_MEMORY_GROWTH")
      klsOptions_.memory_growth = it->getImmutableValue<double>();
    else if (tag == "KLS_HALT_IF_SINGULAR")
      klsOptions_.halt_if_singular = it->getImmutableValue<int>();
    else if (tag == "KLS_FAST_FACTOR")
      klsOptions_.fast_factor = it->getImmutableValue<int>();
    else if (tag == "KLS_STATIC_PIVOTING")
      klsOptions_.static_pivoting = it->getImmutableValue<int>();
    else if (tag == "KLS_EXPECTED_REFACTORIZATIONS")
      klsOptions_.expected_refactorizations = it->getImmutableValue<int>();
    else if (tag == "KLS_EXPECTED_SOLVES")
      klsOptions_.expected_solves = it->getImmutableValue<int>();
  }

  if (klsOptions_.backend == KLS_BACKEND_SERIAL && klsOptions_.threads != 1)
    Report::UserError0() << "KLS_BACKEND=SERIAL requires KLS_THREADS=1";

  delete options_;
  options_ = new Util::OptionBlock(OB);
  clearAnalysis_();

  return true;
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::doSolve
// Purpose       :
// Special Notes :
// Scope         : Public
//-----------------------------------------------------------------------------
int KLSSolver::doSolve(bool reuse_factors, bool transpose)
{
  ProfileTimer profileTimer(profile_ ? &profileStats_.total : 0);
  if (profile_) ++profileStats_.calls;
  timer_->resetStartTime();

  int linearStatus = 0;
  int klsStatus = KLS_OK;
  bool solutionReady = false;
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
      klsStatus = analyze_(prob);

    if (klsStatus == KLS_OK && (!reuse_factors || !factored_))
    {
      if (factored_ && !transpose)
      {
        klsStatus = refactorSolve_(prob);
        solutionReady = (klsStatus == KLS_OK);
        if (klsStatus != KLS_OK)
          klsStatus = factor_(prob);
      }
      else
      {
        klsStatus = factor_(prob);
      }
      if (klsStatus == KLS_ERR_UNSUPPORTED && !analyzed_)
      {
        klsStatus = analyze_(prob);
        if (klsStatus == KLS_OK)
          klsStatus = factor_(prob);
      }
    }

    if (klsStatus == KLS_OK && !solutionReady)
      klsStatus = solve_(prob, transpose);
  }

  int localFailure = (klsStatus == KLS_OK) ? 0 : 1;
  int globalFailure = localFailure;
  if (matrix->Comm().NumProc() > 1)
    matrix->Comm().MaxAll(&localFailure, &globalFailure, 1);

  if (globalFailure)
  {
    linearStatus = -1;
    prob->GetLHS()->PutScalar(0.0);
    if (myPID == 0)
    {
      Report::UserWarning0()
        << "KLS linear solve failed: " << kls_status_string(klsStatus)
        << " (" << klsStatus << "), returning zero solution to nonlinear solver!";
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
    Xyce::lout() << "  KLS Solve Time: " << endSolveTime << std::endl;
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
    Xyce::lout() << "Linear System Residual (KLS): " << std::endl;
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
    Xyce::lout() << "Total Linear Solution Time (KLS): " << solutionTime_ << std::endl;

  return linearStatus;
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::analyze_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int KLSSolver::analyze_(Epetra_LinearProblem * problem)
{
  ProfileTimer profileTimer(profile_ ? &profileStats_.analysis : 0);
  Epetra_CrsMatrix * matrix = dynamic_cast<Epetra_CrsMatrix *>(problem->GetMatrix());
  if (!matrix)
    return KLS_ERR_INVALID_ARGUMENT;

  const int buildStatus = buildCSR_(matrix);
  if (buildStatus != KLS_OK)
    return buildStatus;

  if (!solver_)
  {
    const int createStatus = kls_create(&solver_);
    if (createStatus != KLS_OK)
      return createStatus;
  }

  const int n = matrix->NumMyRows();
  const int status = kls_analyze_csr(solver_, KLS_INDEX_INT32, n,
                                     &rowPtr_[0], colIdx_.empty() ? 0 : &colIdx_[0],
                                     0, &klsOptions_);
  analyzed_ = (status == KLS_OK);
  factored_ = false;
  return status;
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::factor_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int KLSSolver::factor_(Epetra_LinearProblem * problem)
{
  Epetra_CrsMatrix * matrix = dynamic_cast<Epetra_CrsMatrix *>(problem->GetMatrix());
  if (!matrix || !solver_)
    return KLS_ERR_INVALID_ARGUMENT;

  int status = updateValues_(matrix);
  if (status != KLS_OK)
    return status;

  ProfileTimer profileTimer(profile_ ? &profileStats_.factor : 0);

  if (factored_)
  {
    status = kls_refactor(solver_, values_.empty() ? 0 : &values_[0]);
    if (status != KLS_OK)
      status = kls_factor(solver_, values_.empty() ? 0 : &values_[0]);
  }
  else
  {
    status = kls_factor(solver_, values_.empty() ? 0 : &values_[0]);
  }

  factored_ = (status == KLS_OK);
  return status;
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::refactorSolve_
// Purpose       : Refactor and solve one ordinary Xyce right-hand side.
//-----------------------------------------------------------------------------
int KLSSolver::refactorSolve_(Epetra_LinearProblem * problem)
{
  Epetra_CrsMatrix * matrix = dynamic_cast<Epetra_CrsMatrix *>(problem->GetMatrix());
  Epetra_MultiVector * X = problem->GetLHS();
  Epetra_MultiVector * B = problem->GetRHS();
  if (!matrix || !solver_ || !factored_ || !X || !B ||
      X->NumVectors() != 1 || B->NumVectors() != 1)
    return KLS_ERR_UNSUPPORTED;

  bool valuesChanged = false;
  int status = updateValues_(matrix, &valuesChanged);
  if (status != KLS_OK)
    return status;

  double ** rhsPtrs = 0;
  double ** lhsPtrs = 0;
  if (B->ExtractView(&rhsPtrs) != 0 || X->ExtractView(&lhsPtrs) != 0)
    return KLS_ERR_INVALID_ARGUMENT;

  ProfileTimer profileTimer(profile_ ? &profileStats_.refactorSolve : 0);
  status = valuesChanged
    ? kls_refactor_solve(solver_, values_.empty() ? 0 : &values_[0],
                         1, rhsPtrs[0], 0, lhsPtrs[0], 0)
    : kls_solve(solver_, 1, rhsPtrs[0], 0, lhsPtrs[0], 0);
  factored_ = (status == KLS_OK);
  return status;
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::solve_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int KLSSolver::solve_(Epetra_LinearProblem * problem, bool transpose)
{
  ProfileTimer profileTimer(profile_ ? &profileStats_.solve : 0);
  if (!solver_ || !factored_)
    return KLS_ERR_INVALID_ARGUMENT;

  Epetra_MultiVector * X = problem->GetLHS();
  Epetra_MultiVector * B = problem->GetRHS();
  if (!X || !B || X->NumVectors() != B->NumVectors())
    return KLS_ERR_INVALID_ARGUMENT;

  double ** rhsPtrs = 0;
  double ** lhsPtrs = 0;
  if (B->ExtractView(&rhsPtrs) != 0 || X->ExtractView(&lhsPtrs) != 0)
    return KLS_ERR_INVALID_ARGUMENT;

  int status = KLS_OK;
  for (int rhs = 0; rhs < B->NumVectors(); ++rhs)
  {
    if (transpose)
      status = kls_solve_transpose(solver_, 1, rhsPtrs[rhs], 0, lhsPtrs[rhs], 0);
    else
      status = kls_solve(solver_, 1, rhsPtrs[rhs], 0, lhsPtrs[rhs], 0);

    if (status != KLS_OK)
      return status;
  }

  return status;
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::buildCSR_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int KLSSolver::buildCSR_(Epetra_CrsMatrix * matrix)
{
  if (!matrix)
    return KLS_ERR_INVALID_ARGUMENT;

  if (matrix->NumGlobalRows() != matrix->NumGlobalCols() ||
      matrix->NumMyRows() != matrix->NumGlobalRows())
    return KLS_ERR_UNSUPPORTED;

  if (matrix->NumMyRows() < 0 || matrix->NumMyNonzeros() < 0)
    return KLS_ERR_INVALID_ARGUMENT;

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
      return KLS_ERR_INVALID_ARGUMENT;

    rowPtr_[static_cast<std::size_t>(row)] = offset;
    for (int entry = 0; entry < numEntries; ++entry)
    {
      if (indices[entry] < 0 || indices[entry] >= n)
        return KLS_ERR_UNSUPPORTED;
      colIdx_[static_cast<std::size_t>(offset)] = indices[entry];
      values_[static_cast<std::size_t>(offset)] = rowValues[entry];
      ++offset;
    }
  }

  if (offset != nnz)
    return KLS_ERR_INVALID_ARGUMENT;

  rowPtr_[static_cast<std::size_t>(n)] = offset;
  return KLS_OK;
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::updateValues_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int KLSSolver::updateValues_(Epetra_CrsMatrix * matrix, bool * changed)
{
  ProfileTimer profileTimer(profile_ ? &profileStats_.values : 0);
  if (!matrix)
    return KLS_ERR_INVALID_ARGUMENT;

  const int n = matrix->NumMyRows();
  if (static_cast<std::size_t>(matrix->NumMyNonzeros()) != values_.size() ||
      rowPtr_.size() != static_cast<std::size_t>(n) + 1)
  {
    clearAnalysis_();
    return KLS_ERR_UNSUPPORTED;
  }

  bool anyChanged = false;
  int offset = 0;
  for (int row = 0; row < n; ++row)
  {
    int numEntries = 0;
    double * rowValues = 0;
    const int ierr = matrix->ExtractMyRowView(row, numEntries, rowValues);
    if (ierr != 0)
      return KLS_ERR_INVALID_ARGUMENT;

    if (rowPtr_[static_cast<std::size_t>(row)] != offset ||
        rowPtr_[static_cast<std::size_t>(row) + 1] != offset + numEntries)
    {
      clearAnalysis_();
      return KLS_ERR_UNSUPPORTED;
    }

    if (numEntries > 0)
    {
      double * destination = &values_[static_cast<std::size_t>(offset)];
      const std::size_t bytes = static_cast<std::size_t>(numEntries) * sizeof(double);
      if (std::memcmp(destination, rowValues, bytes) != 0)
      {
        anyChanged = true;
        std::memcpy(destination, rowValues, bytes);
      }
      offset += numEntries;
    }
  }

  if (changed)
    *changed = anyChanged;

  return KLS_OK;
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::clearAnalysis_
// Purpose       :
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
void KLSSolver::clearAnalysis_()
{
  if (solver_)
  {
    kls_destroy(solver_);
    solver_ = 0;
  }
  analyzed_ = false;
  factored_ = false;
  rowPtr_.clear();
  colIdx_.clear();
  values_.clear();
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::importToSerial_
// Purpose       : Import a distributed matrix to a serial one for KLS.
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
Epetra_LinearProblem * KLSSolver::importToSerial_()
{
  ProfileTimer profileTimer(profile_ ? &profileStats_.imports : 0);
  usingOriginalProblem_ = false;
#ifdef Xyce_PARALLEL_MPI
  Epetra_CrsMatrix * origMat = dynamic_cast<Epetra_CrsMatrix *>(problem_->GetOperator());

  // Local CSR column indices must address the same ordering as matrix rows
  // and both vectors. A single rank alone does not guarantee this.
  if (origMat->Comm().NumProc() == 1 &&
      origMat->RowMap().SameAs(origMat->ColMap()) &&
      origMat->RowMap().SameAs(origMat->OperatorDomainMap()) &&
      origMat->RowMap().SameAs(origMat->OperatorRangeMap()) &&
      origMat->RowMap().SameAs(problem_->GetLHS()->Map()) &&
      origMat->RowMap().SameAs(problem_->GetRHS()->Map()))
  {
    usingOriginalProblem_ = true;
    if (profile_) ++profileStats_.directCalls;
    return problem_;
  }

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
  usingOriginalProblem_ = true;
  if (profile_) ++profileStats_.directCalls;
  return problem_;
#endif
}

//-----------------------------------------------------------------------------
// Function      : KLSSolver::exportToGlobal_
// Purpose       : Export the serial solution to the global vector.
// Special Notes :
// Scope         : Private
//-----------------------------------------------------------------------------
int KLSSolver::exportToGlobal_()
{
  ProfileTimer profileTimer(profile_ ? &profileStats_.exports : 0);
#ifdef Xyce_PARALLEL_MPI
  if (!usingOriginalProblem_)
    return problem_->GetLHS()->Export(*serialLHS_, *serialImporter_, Insert);
#endif
  return 0;
}

} // namespace Linear
} // namespace Xyce
