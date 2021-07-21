#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  w << 1, 9;
  N_Vector v = create_eigen_nvector<Vector>(&w);
  v->ops->nvconst(3, v);

  auto v_vec = static_cast<Vector*>(v->content);
  std::cout << *v_vec;

}