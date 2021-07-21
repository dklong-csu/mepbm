#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  realtype coeff[] {2, 3, 4};

  // vector to scale
  Vector s(2);
  s << 1, 2;
  N_Vector scale = create_eigen_nvector<Vector>(&s);

  // vectors to be added to
  Vector w(2);
  w << 1, 2;
  N_Vector x = create_eigen_nvector<Vector>(&w);

  Vector v(2);
  v << 2, 3;
  N_Vector y = create_eigen_nvector<Vector>(&v);

  Vector u(2);
  u << 3, 4;
  N_Vector z = create_eigen_nvector<Vector>(&u);

  N_Vector X [3] = {x, y, z};

  // vectors to store results
  Vector a(2);
  N_Vector d = create_eigen_nvector<Vector>(&a);

  Vector b(2);
  N_Vector e = create_eigen_nvector<Vector>(&b);

  Vector c(2);
  N_Vector f = create_eigen_nvector<Vector>(&c);

  N_Vector Y [3] = {d, e, f};

  // test
  auto result = x->ops->nvscaleaddmulti(3, coeff, scale, X, Y);
  std::cout << result << std::endl;
  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;

  result = x->ops->nvscaleaddmulti(0, coeff, scale, X, Y);
  std::cout << result;
}