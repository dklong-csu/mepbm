#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(4);
  w << 2, 1, -1, -2;
  N_Vector c = create_eigen_nvector<Vector>(&w);

  Vector v(4);
  N_Vector x = create_eigen_nvector<Vector>(&v);

  Vector u(4);
  N_Vector m = create_eigen_nvector<Vector>(&u);

  // we expect all to pass
  v << 1, 0, 0, -1;
  booleantype result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << u << std::endl;

  // we expect c=2 to fail
  v << 0, 0, 0, -1;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << u << std::endl;

  // we expect c=1 to fail
  v << 1, -1, 0, -1;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << u << std::endl;

  // we expect c=-1 to fail
  v << 1, 0, 1, -1;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << u << std::endl;

  // we expect c=-2 to fail
  v << 1, 0, 0, 0;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << u << std::endl;

  // we expect all to fail
  v << 0, -1, 1, 0;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << u << std::endl;


}