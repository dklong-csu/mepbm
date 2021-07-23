#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector x = create_eigen_nvector<Vector>(4);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 1, 2, 4, 8;

  N_Vector w = create_eigen_nvector<Vector>(4);
  auto w_vec = static_cast<Vector*>(w->content);
  *w_vec << 2, 1, .5, .25;


  // Result should be 2
  auto result = x->ops->nvwrmsnorm(x,w);
  std::cout << result;
}