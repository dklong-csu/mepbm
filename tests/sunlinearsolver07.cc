#include "src/create_sunmatrix.h"
#include "src/create_nvector.h"
#include "src/create_sunlinearsolver.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <iostream>
#include <iomanip>


using Matrix = Eigen::SparseMatrix<double>; // column- or row-major is fine
using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Solver = Eigen::BiCGSTAB< Matrix, Eigen::IncompleteLUT<double> >; // IncompleteLUT is a preconditioner and appears to be the best default setting


int main ()
{
  // Create a matrix
  auto A = MEPBM::create_eigen_sunmatrix<Matrix>(10, 10);
  auto M = static_cast<Matrix*>(A->content);
  for (unsigned int i=0; i<10;++i)
  {
    // Diagonal = 2
    M->coeffRef(i,i) = 2;
    // Off diagonal = -1
    if (i<9)
    {
      M->coeffRef(i,i+1) = -1;
    }
    if (i>0)
    {
      M->coeffRef(i,i-1) = -1;
    }
  }


  // Create the right-hand side vector
  auto b = MEPBM::create_eigen_nvector<Vector>(10);
  auto vec = static_cast<Vector*>(b->content);
  *vec << 8, 4, 6, 2, 7, 3, 7, 7, 8, 5;

  // Create the linear solver
  auto solver = MEPBM::create_sparse_iterative_solver<Matrix, double, Solver>();
  solver->ops->setup(solver, A);

  // Create the solution vector
  auto x = MEPBM::create_eigen_nvector<Vector>(10);

  // Solve for x
  solver->ops->solve(solver, A, x, b, 1e-7);

  // output solution for comparison
  // output confirmed by solving the same system in Matlab
  auto x_vec = *static_cast<Vector*>(x->content);
  std::cout << std::setprecision(20) << x_vec << std::endl;
}