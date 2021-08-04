#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  realtype coeff[] {2, 3, 4};

  // vector to scale
  N_Vector scale = create_eigen_nvector<Vector>(2);
  auto scale_vec = static_cast<Vector*>(scale->content);
  *scale_vec << 1, 2;


  // vectors to be added to
  N_Vector x = create_eigen_nvector<Vector>(2);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 1,2;


  N_Vector y = create_eigen_nvector<Vector>(2);
  auto y_vec = static_cast<Vector*>(y->content);
  *y_vec << 2,3;


  N_Vector z = create_eigen_nvector<Vector>(2);
  auto z_vec = static_cast<Vector*>(z->content);
  *z_vec << 3, 4;


  N_Vector X [3] = {x, y, z};

  // vectors to store results
  N_Vector a = create_eigen_nvector<Vector>(2);

  N_Vector b = create_eigen_nvector<Vector>(2);

  N_Vector c = create_eigen_nvector<Vector>(2);

  N_Vector Y [3] = {a, b, c};

  // test
  auto result = x->ops->nvscaleaddmulti(3, coeff, scale, X, Y);
  std::cout << result << std::endl;
  std::cout << *(static_cast<Vector*>(a->content)) << std::endl;
  std::cout << *(static_cast<Vector*>(b->content)) << std::endl;
  std::cout << *(static_cast<Vector*>(c->content)) << std::endl;

  result = x->ops->nvscaleaddmulti(0, coeff, scale, X, Y);
  std::cout << result << std::endl;
}