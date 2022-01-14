#ifndef MEPBM_CREATE_SUNLINEARSOLVER_H
#define MEPBM_CREATE_SUNLINEARSOLVER_H

#include "sunlinearsolver_operations.h"



namespace MEPBM {
  /// Function to create a SUNDIALS linear solver using Eigen as the backend -- for iterative solvers on sparse matrices
  template<typename MatrixType, typename Real, typename SolverType>
  SUNLinearSolver
  create_sparse_iterative_solver()
  {
    // create empty linear solver
    SUNLinearSolver solver = SUNLinSolNewEmpty();

    // Attach operations
    solver->ops->gettype = MEPBM::get_type_iterative;
    solver->ops->setup = MEPBM::setup_sparse_solver<MatrixType, SolverType>;
    solver->ops->solve = MEPBM::solve<MatrixType, Real, SolverType>;
    solver->ops->free = MEPBM::free<SolverType>;

    // Attach content
    SolverType* eigen_solver
        = new SolverType();

    solver->content = (void*)eigen_solver;

    return solver;
  }



  /// Function to create a SUNDIALS linear solver using Eigen as the backend -- for direct solvers on sparse matrices
  template<typename MatrixType, typename Real, typename SolverType>
  SUNLinearSolver
  create_sparse_direct_solver()
  {
    // create empty linear solver
    SUNLinearSolver solver = SUNLinSolNewEmpty();

    // Attach operations
    solver->ops->gettype = MEPBM::get_type_direct;
    solver->ops->setup = MEPBM::setup_sparse_solver<MatrixType, SolverType>;
    solver->ops->solve = MEPBM::solve<MatrixType, Real, SolverType>;
    solver->ops->free = MEPBM::free<SolverType>;

    // Attach content
    SolverType* eigen_solver
        = new SolverType();

    solver->content = (void*)eigen_solver;

    return solver;
  }


/*
 * There aren't actually any dense iterative solvers in Eigen, but here is the creation function if there ever are.
  /// Function to create a SUNDIALS linear solver using Eigen as the backend -- for iterative solvers on dense matrices
  template<typename MatrixType, typename Real, typename SolverType>
  SUNLinearSolver
  create_dense_iterative_solver()
  {
    // create empty linear solver
    SUNLinearSolver solver = SUNLinSolNewEmpty();

    // Attach operations
    solver->ops->gettype = MEPBM::get_type_iterative;
    solver->ops->setup = MEPBM::setup_dense_solver<MatrixType, SolverType>;
    solver->ops->solve = MEPBM::solve<MatrixType, Real, SolverType>;
    solver->ops->free = MEPBM::free<SolverType>;

    // Attach content
    SolverType* eigen_solver
        = new SolverType();

    solver->content = (void*)eigen_solver;

    return solver;
  }
*/


  /// Function to create a SUNDIALS linear solver using Eigen as the backend -- for direct solvers on dense matrices
  template<typename MatrixType, typename Real, typename SolverType>
  SUNLinearSolver
  create_dense_direct_solver()
  {
    // create empty linear solver
    SUNLinearSolver solver = SUNLinSolNewEmpty();

    // Attach operations
    solver->ops->gettype = MEPBM::get_type_direct;
    solver->ops->setup = MEPBM::setup_dense_solver<MatrixType, SolverType>;
    solver->ops->solve = MEPBM::solve<MatrixType, Real, SolverType>;
    solver->ops->free = MEPBM::free<SolverType>;

    // Attach content
    SolverType* eigen_solver
        = new SolverType();

    solver->content = (void*)eigen_solver;

    return solver;
  }


}

#endif //MEPBM_CREATE_SUNLINEARSOLVER_H
