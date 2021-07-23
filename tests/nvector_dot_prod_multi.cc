#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  // vector to dot with others
  N_Vector x = create_eigen_nvector<Vector>(2);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 1, 2;

  // vectors to be dotted with
  N_Vector y = create_eigen_nvector<Vector>(2);
  auto y_vec = static_cast<Vector*>(y->content);
  *y_vec << 2, 3;

  N_Vector z = create_eigen_nvector<Vector>(2);
  auto z_vec = static_cast<Vector*>(z->content);
  *z_vec << 3, 4;

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