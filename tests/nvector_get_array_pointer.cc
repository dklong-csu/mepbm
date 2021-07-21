#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  N_Vector v = create_eigen_nvector<Vector>(&w);

  auto eigen_vec = static_cast<Vector *>(v->content);
  auto eigen_ptr = eigen_vec->data();
  auto fcn_result = v->ops->nvgetarraypointer(v);
  std::cout << (fcn_result == eigen_ptr);
}