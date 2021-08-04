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



  SUNMatrix B = create_eigen_sunmatrix<Matrix>(2,2);
  SUNMatrix C = create_empty_eigen_sunmatrix<Matrix>();

  

  // test the matrix with initialized memory
  {
    auto ierr = A->ops->copy(A, B);
    std::cout << ierr << std::endl;

    auto mat = *static_cast<Matrix*>(B->content);

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

  // test the matrix with uninitialized memory
  {
    auto ierr = A->ops->copy(A, C);
    std::cout << ierr << std::endl;

    auto mat = *static_cast<Matrix*>(C->content);

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