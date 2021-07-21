#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  w << 2, 4;
  N_Vector x = create_eigen_nvector<Vector>(&w);

  Vector u(2);
  N_Vector y = create_eigen_nvector<Vector>(&u);

  // pass the test
  auto result = x->ops->nvinvtest(x,y);
  std::cout << result << std::endl;
  std::cout << u << std::endl;

  // fail the test
  w << 0, 4;
  result = x->ops->nvinvtest(x,y);
  std::cout << result;

}