#include "src/create_nvector.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector v = MEPBM::create_eigen_nvector<Vector>(2);
  auto v_vec = static_cast<Vector*>(v->content);
  *v_vec << -2, 1;


  auto m = v->ops->nvmin(v);
  std::cout << m << std::endl;

  v->ops->nvdestroy(v);
}