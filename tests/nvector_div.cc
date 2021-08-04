#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector x = create_eigen_nvector<Vector>(2);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 4, 16;


  N_Vector y = create_eigen_nvector<Vector>(2);
  auto y_vec = static_cast<Vector*>(y->content);
  *y_vec << 2, 4;

  N_Vector z = create_eigen_nvector<Vector>(2);

  // z = x ./ y = (4/2, 16/4) = (2, 4)
  x->ops->nvdiv(x,y,z);

  auto z_vec = static_cast<Vector*>(z->content);
  std::cout << *z_vec << std::endl;
}