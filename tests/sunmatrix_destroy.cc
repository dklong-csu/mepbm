#include "sunmatrix_eigen.h"
#include "eigen3/Eigen/Sparse"
#include <iostream>

using Matrix = Eigen::SparseMatrix<realtype>;

int main ()
{
  SUNMatrix A = create_eigen_sunmatrix<Matrix>(2,2);

  A->ops->destroy(A);
  std::cout << std::boolalpha << (A->ops == nullptr) << std::endl;
  std::cout << std::boolalpha << (A->content == nullptr) << std::endl;
}