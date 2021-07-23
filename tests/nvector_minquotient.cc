#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector num = create_eigen_nvector<Vector>(3);
  auto num_vec = static_cast<Vector*>(num->content);
  *num_vec << 1, 2, 4;


  N_Vector denom1 = create_eigen_nvector<Vector>(3);
  auto denom1_vec = static_cast<Vector*>(denom1->content);
  *denom1_vec << 4, 2, 1;


  N_Vector denom2 = create_eigen_nvector<Vector>(3);
  auto denom2_vec = static_cast<Vector*>(denom2->content);
  *denom2_vec << 0, 0, 0;


  auto mq1 = num->ops->nvminquotient(num, denom1);
  auto mq2 = num->ops->nvminquotient(num, denom2);

  std::cout << mq1 << std::endl;
  std::cout << mq2 << std::endl;
}