#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  w << -1, 2;
  N_Vector v = create_eigen_nvector<Vector>(&w);
  N_Vector x = v->ops->nvclone(v);
  v->ops->nvabs(v, x);

  auto vec = *(static_cast<Vector *>(x->content));
  std::cout << vec;
}