#ifndef MEPBM_LINEAR_SOLVER_EIGEN_H
#define MEPBM_LINEAR_SOLVER_EIGEN_H

#include "sundials/sundials_linearsolver.h"
#include "eigen3/Eigen/Sparse"



/// Enumeration for (sparse) linear solver type -- only BiCGSTAB is implemented, but this is used in case more functionality is added.
enum EigenLinSolType {BICGSTAB};



namespace SUNLinearSolverOperations
{
  /// Returns the type identifier of the linear solver -- currently only sparse, iterative, matrix methods are supported.
  SUNLinearSolver_Type
  get_type(SUNLinearSolver solver)
  {
    return SUNLINEARSOLVER_MATRIX_ITERATIVE;
  }



  /// Performs linear solver setup based on an updated matrix A. This also sets the preconditioner.
  template<typename MatrixType, typename Real>
  int
  setup_solver(SUNLinearSolver solver, SUNMatrix A)
  {
    auto eigen_solver = static_cast< Eigen::BiCGSTAB< MatrixType, Eigen::IncompleteLUT<Real> >* >(solver->content);
    auto mat = static_cast< MatrixType* >(A->content);

    eigen_solver->compute(*mat);

    return 0;
  }



  /// Solves a linear system Ax=b
  template<typename MatrixType, typename Real>
  int
  solve(SUNLinearSolver solver, SUNMatrix A, N_Vector x, N_Vector b, Real tol)
  {
    auto eigen_solver = static_cast< Eigen::BiCGSTAB< MatrixType, Eigen::IncompleteLUT<Real> >* >(solver->content);
    auto x_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);
    auto b_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(b->content);

    // The `setup' function preps `eigen_solver' with the matrix A, so it does not need to be used here.
    eigen_solver->setTolerance(tol);
    *x_vec = eigen_solver->solve(*b_vec);

    return 0;
  }



  /// Frees memory allocated by the linear solver
  template<typename MatrixType, typename Real>
  int
  free(SUNLinearSolver solver)
  {
    // solver->content will be on heap. delete it and set content to nullptr.
    auto eigen_solver = static_cast< Eigen::BiCGSTAB< MatrixType, Eigen::IncompleteLUT<Real> >* >(solver->content);
    delete eigen_solver;
    solver->content = nullptr;

    // sundials then provides routine to free an empty linear solver
    SUNLinSolFreeEmpty(solver);

    return 0;
  }
}



/// Function to create a linear solver using Eigen software for linear algebra
template<typename MatrixType, typename Real>
SUNLinearSolver
create_eigen_linear_solver()
{
  // create empty linear solver
  SUNLinearSolver solver = SUNLinSolNewEmpty();

  // Attach operations
  solver->ops->gettype = SUNLinearSolverOperations::get_type;
  solver->ops->setup = SUNLinearSolverOperations::setup_solver<MatrixType, Real>;
  solver->ops->solve = SUNLinearSolverOperations::solve<MatrixType, Real>;
  solver->ops->free = SUNLinearSolverOperations::free<MatrixType, Real>;

  // Attach content
  Eigen::BiCGSTAB< MatrixType, Eigen::IncompleteLUT<Real> > *eigen_solver
  = new Eigen::BiCGSTAB< MatrixType, Eigen::IncompleteLUT<Real> >;

  solver->content = (void*)eigen_solver;

  return solver;
}


#endif //MEPBM_LINEAR_SOLVER_EIGEN_H
