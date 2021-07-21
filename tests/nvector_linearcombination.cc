#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  realtype c[] {2, 3, 4};

  Vector w(2);
  w << 1, 2;
  N_Vector x = create_eigen_nvector<Vector>(&w);

  Vector v(2);
  v << 2, 3;
  N_Vector y = create_eigen_nvector<Vector>(&v);

  Vector u(2);
  u << 3, 4;
  N_Vector z = create_eigen_nvector<Vector>(&u);

  N_Vector X [3] = {x, y, z};

  Vector t(2);
  N_Vector r = create_eigen_nvector<Vector>(&t);

  auto result = x->ops->nvlinearcombination(3, c, X, r);
  std::cout << result << std::endl;
  std::cout << t << std::endl;

  result = x->ops->nvlinearcombination(0, c, X, r);
  std::cout << result;
}