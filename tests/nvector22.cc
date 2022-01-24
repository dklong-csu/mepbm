#include "src/create_nvector.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  // constants to scale by
  realtype c [2] = {2,3};


  // vectors to be scaled
  N_Vector y = MEPBM::create_eigen_nvector<Vector>(2);
  auto y_vec = static_cast<Vector*>(y->content);
  *y_vec << 1,2;


  N_Vector z = MEPBM::create_eigen_nvector<Vector>(2);
  auto z_vec = static_cast<Vector*>(z->content);
  *z_vec << 3,4;


  N_Vector X [2] = {y,z};


  // vectors to store results
  N_Vector a = MEPBM::create_eigen_nvector<Vector>(2);

  N_Vector b = MEPBM::create_eigen_nvector<Vector>(2);

  N_Vector C [2] = {a,b};

  // test
  auto result = y->ops->nvscalevectorarray(2, c, X, C);
  std::cout << result << std::endl;
  std::cout << *(static_cast<Vector*>(a->content)) << std::endl;
  std::cout << *(static_cast<Vector*>(b->content)) << std::endl;

  result = y->ops->nvscalevectorarray(0, c, X, C);
  std::cout << result << std::endl;
}