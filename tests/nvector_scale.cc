#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector x = create_eigen_nvector<Vector>(2);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 2, 3;

  realtype c = 5;

  N_Vector y = create_eigen_nvector<Vector>(2);

  x->ops->nvscale(c, x, y);
  auto y_vec = static_cast<Vector*>(y->content);
  std::cout << *y_vec;

}