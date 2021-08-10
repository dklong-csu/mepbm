#include "sunmatrix_eigen.h"
#include "eigen3/Eigen/Sparse"
#include <iostream>

using Matrix = Eigen::SparseMatrix<realtype>;

int main ()
{
  SUNMatrix A = create_eigen_sunmatrix<Matrix>(2,2);
  auto A_matrix = static_cast<Matrix*>(A->content);
  A_matrix->insert(0,0) = 1;
  A_matrix->insert(1,0) = 2;

  // make sure the matrix is non-zero first
  {
    auto mat = *static_cast<Matrix*>(A->content);

    auto rows = mat.rows();
    auto cols = mat.cols();

    for (unsigned int i=0; i<rows; ++i)
      for (unsigned int j=0; j<cols; ++j)
      {
        std::cout << mat.coeff(i,j);
        if (j < cols-1)
          std::cout << ", ";
        else
          std::cout << std::endl;
      }
  }


  A->ops->zero(A);
  // make sure the matrix is now zero
  {
    auto mat = *static_cast<Matrix*>(A->content);

    auto rows = mat.rows();
    auto cols = mat.cols();

    for (unsigned int i=0; i<rows; ++i)
      for (unsigned int j=0; j<cols; ++j)
      {
        std::cout << mat.coeff(i,j);
        if (j < cols-1)
          std::cout << ", ";
        else
          std::cout << std::endl;
      }
  }
}