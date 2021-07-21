#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  N_Vector v = create_eigen_nvector<Vector>(&w);

  N_Vector x = v->ops->nvclone(v);
  // FIXME I'm not sure how to test this
  std::cout << "help plz";
}