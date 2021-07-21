#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  w << 4, 16;
  N_Vector x = create_eigen_nvector<Vector>(&w);

  Vector v(2);
  v << 2, 4;
  N_Vector y = create_eigen_nvector<Vector>(&v);

  Vector t(2);
  N_Vector z = create_eigen_nvector<Vector>(&t);

  // z = x ./ y = (4/2, 16/4) = (2, 4)
  x->ops->nvdiv(x,y,z);

  auto z_vec = static_cast<Vector*>(z->content);
  std::cout << *z_vec;
}