#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector v(4);
  v << 1, 2, 4, 8;
  N_Vector x = create_eigen_nvector<Vector>(&v);

  Vector u(4);
  u << 2, 1, .5, .25;
  N_Vector w = create_eigen_nvector<Vector>(&u);

  // Result should be 2
  auto result = x->ops->nvwrmsnorm(x,w);
  std::cout << result;
}