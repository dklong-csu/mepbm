#include "src/create_nvector.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector x = MEPBM::create_eigen_nvector<Vector>(2);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 2, 3;

  N_Vector y = MEPBM::create_eigen_nvector<Vector>(2);
  auto y_vec = static_cast<Vector*>(y->content);
  *y_vec << 4, 5;

  N_Vector z = MEPBM::create_eigen_nvector<Vector>(2);

  // z = x .* y = (2*4, 3*5) = (8, 15)
  x->ops->nvprod(x,y,z);

  auto z_vec = static_cast<Vector*>(z->content);
  std::cout << *z_vec << std::endl;
}