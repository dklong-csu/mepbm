#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  w << 2, 3;
  N_Vector x = create_eigen_nvector<Vector>(&w);

  realtype c = 5;

  Vector v(2);
  N_Vector y = create_eigen_nvector<Vector>(&v);

  x->ops->nvscale(c, x, y);
  auto y_vec = static_cast<Vector*>(y->content);
  std::cout << *y_vec;

}