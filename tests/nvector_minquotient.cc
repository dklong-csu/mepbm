#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(3);
  w << 1, 2, 4;
  N_Vector num = create_eigen_nvector<Vector>(&w);

  Vector u(3);
  u << 4, 2, 1;
  N_Vector denom1 = create_eigen_nvector<Vector>(&u);

  Vector v(3);
  v << 0, 0, 0;
  N_Vector denom2 = create_eigen_nvector<Vector>(&v);


  auto mq1 = num->ops->nvminquotient(num, denom1);
  auto mq2 = num->ops->nvminquotient(num, denom2);

  std::cout << mq1 << std::endl;
  std::cout << mq2 << std::endl;
}