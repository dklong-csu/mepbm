#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  w << -2, 1;
  N_Vector v = create_eigen_nvector<Vector>(&w);

  auto m = v->ops->nvmin(v);
  std::cout << m;
}