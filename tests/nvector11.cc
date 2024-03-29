#include "src/create_nvector.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector v = MEPBM::create_eigen_nvector<Vector>(2);

  sunindextype n = v->ops->nvgetlength(v);
  std::cout << n << std::endl;

  v->ops->nvdestroy(v);
}