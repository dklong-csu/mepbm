#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  // constants to scale by
  realtype c [2] = {2,3};

  // vectors to be scaled
  Vector v(2);
  v << 1,2;
  N_Vector y = create_eigen_nvector<Vector>(&v);

  Vector u(2);
  u << 3,4;
  N_Vector z = create_eigen_nvector<Vector>(&u);

  N_Vector X [2] = {y,z};

  // vectors to store results
  Vector w(2);
  N_Vector a = create_eigen_nvector<Vector>(&w);

  Vector t(2);
  N_Vector b = create_eigen_nvector<Vector>(&t);

  N_Vector C [2] = {a,b};

  // test
  auto result = y->ops->nvscalevectorarray(2, c, X, C);
  std::cout << result << std::endl;
  std::cout << w << std::endl;
  std::cout << t << std::endl;

  result = y->ops->nvscalevectorarray(0, c, X, C);
  std::cout << result;
}