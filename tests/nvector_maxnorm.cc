#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  w << -5, 2;
  N_Vector x = create_eigen_nvector<Vector>(&w);

  realtype m = x->ops->nvmaxnorm(x);
  std::cout << m;
}