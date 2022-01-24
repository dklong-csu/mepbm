#include "src/create_sunmatrix.h"
#include "eigen3/Eigen/Sparse"
#include <iostream>

using Matrix = Eigen::SparseMatrix<realtype>;

int main ()
{
	SUNMatrix A = MEPBM::create_eigen_sunmatrix<Matrix>(2,2);

  auto B = A->ops->clone(A);

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