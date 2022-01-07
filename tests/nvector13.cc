#include "src/create_nvector.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector x = MEPBM::create_eigen_nvector<Vector>(2);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 2, 4;


  N_Vector y = MEPBM::create_eigen_nvector<Vector>(2);

  // pass the test
  auto result = x->ops->nvinvtest(x,y);
  std::cout << result << std::endl;
  std::cout << *static_cast<Vector*>(y->content) << std::endl;

  // fail the test
  *x_vec << 0, 4;
  result = x->ops->nvinvtest(x,y);
  std::cout << result << std::endl;

}