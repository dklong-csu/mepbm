#ifndef MEPBM_LINEAR_SOLVER_EIGEN_H
#define MEPBM_LINEAR_SOLVER_EIGEN_H

#include "sundials/sundials_linearsolver.h"
#include "eigen3/Eigen/Sparse"
#include "eigen3/Eigen/Dense"
#include <iostream>



/// Enumeration for (sparse) linear solver type -- only BiCGSTAB is implemented, but this is used in case more functionality is added.
enum LinearSolverClass {DIRECT, ITERATIVE};



namespace SUNLinearSolverOperations
{
  /// Returns the type identifier of the linear solver when an iterative solver is used.
  SUNLinearSolver_Type
  get_type_iterative(SUNLinearSolver solver)
  {
    return SUNLINEARSOLVER_MATRIX_ITERATIVE;
  }



  /// Returns the type identifier of the linear solver when a direct solver is used.
  SUNLinearSolver_Type
  get_type_direct(SUNLinearSolver solver)
  {
    return SUNLINEARSOLVER_DIRECT;
  }



  /// Performs linear solver setup based on an updated matrix A. This also sets the preconditioner.
  template<typename MatrixType, typename Real, typename SolverType>
  int
  setup_solver(SUNLinearSolver solver, SUNMatrix A)
  {
    auto eigen_solver = static_cast< SolverType* >(solver->content);
    auto mat = static_cast< MatrixType* >(A->content);

    eigen_solver->compute(*mat);

    return 0;
  }



  /// Solves a linear system Ax=b with an iterative solver
  template<typename MatrixType, typename Real, typename SolverType>
  int
  solve_iterative(SUNLinearSolver solver, SUNMatrix A, N_Vector x, N_Vector b, Real tol)
  {
    auto eigen_solver = static_cast< SolverType* >(solver->content);
    auto x_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);
    auto b_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(b->content);

    // The `setup' function preps `eigen_solver' with the matrix A, so it does not need to be used here.
    eigen_solver->setTolerance(tol);
    *x_vec = eigen_solver->solve(*b_vec);

    return 0;
  }



  /// Solves a linear system Ax=b with an iterative solver
  template<typename MatrixType, typename Real, typename SolverType>
  int
  solve_direct(SUNLinearSolver solver, SUNMatrix A, N_Vector x, N_Vector b, Real tol)
  {
    auto eigen_solver = static_cast< SolverType* >(solver->content);
    auto x_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);
    auto b_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(b->content);

    // The `setup' function preps `eigen_solver' with the matrix A, so it does not need to be used here.
    *x_vec = eigen_solver->solve(*b_vec);

    return 0;
  }



  /// Frees memory allocated by the linear solver
  template<typename MatrixType, typename Real, typename SolverType>
  int
  free(SUNLinearSolver solver)
  {
    // solver->content will be on heap. delete it and set content to nullptr.
    auto eigen_solver = static_cast< SolverType* >(solver->content);
    delete eigen_solver;
    solver->content = nullptr;

    // sundials then provides routine to free an empty linear solver
    SUNLinSolFreeEmpty(solver);

    return 0;
  }
}



/**
 * Function to create a linear solver using Eigen software for linear algebra -- iterative solvers
 * Tested combinations of MatrixType and SolverType
 *
 * Eigen::SparseMatrix<Real, Eigen::RowMajor> + Eigen::BiCGSTAB< Matrix, Eigen::IncompleteLUT<double> > -- tested, works.
 *
 *
 */
template<typename MatrixType, typename Real, typename SolverType>
SUNLinearSolver
create_eigen_iterative_linear_solver()
{
  // create empty linear solver
  SUNLinearSolver solver = SUNLinSolNewEmpty();

  // Attach operations
  solver->ops->gettype = SUNLinearSolverOperations::get_type_iterative;
  solver->ops->setup = SUNLinearSolverOperations::setup_solver<MatrixType, Real, SolverType>;
  solver->ops->solve = SUNLinearSolverOperations::solve_iterative<MatrixType, Real, SolverType>;
  solver->ops->free = SUNLinearSolverOperations::free<MatrixType, Real, SolverType>;

  // Attach content
  SolverType* eigen_solver
  = new SolverType();

  solver->content = (void*)eigen_solver;

  return solver;
}



/**
 * Function to create a linear solver using Eigen software for linear algebra -- direct solvers
 * Tested combinations of MatrixType and SolverType
 *
 * Eigen::SparseMatrix<Real, Eigen::ColMajor> + Eigen::SparseLU<Matrix> -- tested, works.
 * Eigen::SparseMatrix<Real, ...> + Eigen::SparseQR<Matrix, ...> -- tested, works.
 * Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> + Eigen::PartialPivLU<Matrix> -- tested, works.
 * Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> + Eigen::FullPivLU<Matrix> -- tested, works.
 * Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> + Eigen::HouseholderQR<Matrix> -- tested, works.
 * Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> + Eigen::ColPivHouseholderQR<Matrix> -- tested, works.
 * Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> + Eigen::FullPivHouseholderQR<Matrix> -- tested, works.
 * Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> + Eigen::CompleteOrthogonalDecomposition<Matrix> -- tested, works.
 *
 */
template<typename MatrixType, typename Real, typename SolverType>
SUNLinearSolver
create_eigen_direct_linear_solver()
{
  // create empty linear solver
  SUNLinearSolver solver = SUNLinSolNewEmpty();

  // Attach operations
  solver->ops->gettype = SUNLinearSolverOperations::get_type_direct;
  solver->ops->setup = SUNLinearSolverOperations::setup_solver<MatrixType, Real, SolverType>;
  solver->ops->solve = SUNLinearSolverOperations::solve_direct<MatrixType, Real, SolverType>;
  solver->ops->free = SUNLinearSolverOperations::free<MatrixType, Real, SolverType>;

  // Attach content
  SolverType* eigen_solver
      = new SolverType();

  solver->content = (void*)eigen_solver;

  return solver;
}



/**
 * Wrapper function to create an iterative or direct solver as appropriate.
 */
 template<typename MatrixType, typename Real, typename SolverType, LinearSolverClass SolverClass>
 SUNLinearSolver
 create_eigen_linear_solver()
{
   if (SolverClass == DIRECT)
   {
     return create_eigen_direct_linear_solver<MatrixType, Real, SolverType>();
   }
   else if (SolverClass == ITERATIVE)
   {
     return create_eigen_iterative_linear_solver<MatrixType, Real, SolverType>();
   }
   else
   {
     std::cerr << "Invalid type of linear solver. Specify either DENSE or ITERATIVE." << std::endl;
   }
}
#endif //MEPBM_LINEAR_SOLVER_EIGEN_H
