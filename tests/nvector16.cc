#include "src/create_nvector.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector x = MEPBM::create_eigen_nvector<Vector>(2);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << -5, 2;


  realtype m = x->ops->nvmaxnorm(x);
  std::cout << m << std::endl;

  x->ops->nvdestroy(x);
}