#include "src/finite_difference.h"
#include <eigen3/Eigen/Dense>
#include <iostream>


// Test backward difference on line

using Real = double;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;


Real
f(const Vector & x)
{
  // y =x_1 + 2x_2 + 3x_3 + ... + ix_i + ...
  Real y = 0;
  for (unsigned int i=0; i<x.size(); ++i)
    y += (i+1)*x(i);
  return y;
}

int main()
{
  Vector x(3);
  x << 1, 10, 100;
  Vector h(3);
  h << 0.1, 0.1, 0.1;

  Vector grad_f = MEPBM::finite_difference_one_sided<Real, Vector>(&f, x, -h);

  // Derivative should be [1, 2, 3]
  std::cout << grad_f << std::endl;
}