#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  realtype c = 1;

  Vector w(3);
  w << 0, 1, 2;
  N_Vector x = create_eigen_nvector<Vector>(&w);

  Vector u(3);
  N_Vector y = create_eigen_nvector<Vector>(&u);

  x->ops->nvcompare(c,x,y);
  std::cout << u;
}