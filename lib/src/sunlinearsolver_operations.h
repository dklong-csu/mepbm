#ifndef MEPBM_SUNLINEARSOLVER_OPERATIONS_H
#define MEPBM_SUNLINEARSOLVER_OPERATIONS_H

#include "sundials/sundials_linearsolver.h"
#include <eigen3/Eigen/Dense>



namespace MEPBM {
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



  /// Performs linear solver setup based on an updated dense matrix A.
  template<typename MatrixType, typename SolverType>
  int
  setup_dense_solver(SUNLinearSolver solver, SUNMatrix A)
  {
    auto eigen_solver = static_cast< SolverType* >(solver->content);
    auto mat = static_cast< MatrixType* >(A->content);

    eigen_solver->compute(*mat);

    return 0;
  }



  /// Performs linear solver setup based on an updated sparse matrix A.
  template<typename MatrixType, typename SolverType>
  int
  setup_sparse_solver(SUNLinearSolver solver, SUNMatrix A)
  {
    auto eigen_solver = static_cast< SolverType* >(solver->content);
    auto mat = static_cast< MatrixType* >(A->content);
    // Solvers in Eigen require the matrix to be in compressed format
    mat->makeCompressed();

    eigen_solver->compute(*mat);

    return 0;
  }



  /// Solves a linear system Ax=b with an iterative solver
  template<typename MatrixType, typename Real, typename SolverType>
  int
  solve(SUNLinearSolver solver, SUNMatrix A, N_Vector x, N_Vector b, Real tol)
  {
    auto eigen_solver = static_cast< SolverType* >(solver->content);
    auto x_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);
    auto b_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(b->content);

    // The `setup' function preps `eigen_solver' with the matrix A, so it does not need to be used here.
    *x_vec = eigen_solver->solve(*b_vec);

    return 0;
  }



  /// Frees memory allocated by the linear solver
  template<typename SolverType>
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

#endif //MEPBM_SUNLINEARSOLVER_OPERATIONS_H
