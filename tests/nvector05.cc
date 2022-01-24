#include "src/create_nvector.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  realtype c = 1;

  N_Vector x = MEPBM::create_eigen_nvector<Vector>(3);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 0, 1, 2;

  N_Vector y = MEPBM::create_eigen_nvector<Vector>(3);

  x->ops->nvcompare(c,x,y);
  std::cout << *static_cast<Vector*>(y->content) << std::endl;
}