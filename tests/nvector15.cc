#include "src/create_nvector.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  realtype a = 2;
  realtype b = 3;


  N_Vector x = MEPBM::create_eigen_nvector<Vector>(2);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 1, 2;

  N_Vector y = MEPBM::create_eigen_nvector<Vector>(2);
  auto y_vec = static_cast<Vector*>(y->content);
  *y_vec << 3, 4;

  N_Vector z = MEPBM::create_eigen_nvector<Vector>(2);

  // z = ax + by
  // z = 2*(1,2) + 3*(3,4) = (11, 16)
  x->ops->nvlinearsum(a,x,b,y,z);
  auto z_vec = static_cast<Vector*>(z->content);
  std::cout << *z_vec << std::endl;

  x->ops->nvdestroy(x);
  y->ops->nvdestroy(y);
  z->ops->nvdestroy(z);
}