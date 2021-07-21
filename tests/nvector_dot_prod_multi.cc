#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  // vector to dot with others
  Vector w(2);
  w << 1,2;
  N_Vector x = create_eigen_nvector<Vector>(&w);

  // vectors to be dotted with
  Vector v(2);
  v << 2,3;
  N_Vector y = create_eigen_nvector<Vector>(&v);

  Vector u(2);
  u << 3,4;
  N_Vector z = create_eigen_nvector<Vector>(&u);

  N_Vector X [2] = {y,z};

  // array to store results
  realtype d[2];

  // test
  auto result = x->ops->nvdotprodmulti(2, x, X, d);
  std::cout << result << std::endl;
  std::cout << d[0] << std::endl;
  std::cout << d[1] << std::endl;

  result = x->ops->nvdotprodmulti(0, x, X, d);
  std::cout << result;
}