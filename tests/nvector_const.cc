#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector v = create_eigen_nvector<Vector>(2);
  auto v_vec = static_cast<Vector*>(v->content);
  *v_vec << 1, 9;

  v->ops->nvconst(3, v);

  std::cout << *v_vec << std::endl;

}