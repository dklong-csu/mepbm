#include "sunmatrix_eigen.h"
#include "nvector_eigen.h"
#include "linear_solver_eigen.h"

#include <iostream>
#include <iomanip>


using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;


int main ()
{
  // Create a matrix
  auto A = create_eigen_sunmatrix<Matrix>(10, 10);
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
  M->makeCompressed();

  // Create the right-hand side vector
  auto b = create_eigen_nvector<Vector>(10);
  auto vec = static_cast<Vector*>(b->content);
  *vec << 8, 4, 6, 2, 7, 3, 7, 7, 8, 5;

  // Create the linear solver
  auto solver = create_eigen_linear_solver<Matrix, double>();
  solver->ops->setup(solver, A);

  // Create the solution vector
  auto x = create_eigen_nvector<Vector>(10);

  // Solve for x
  solver->ops->solve(solver, A, x, b, 1e-7);

  // output solution for comparison
  // output confirmed by solving the same system in Matlab
  auto x_vec = *static_cast<Vector*>(x->content);
  std::cout << std::setprecision(20) << x_vec << std::endl;
}