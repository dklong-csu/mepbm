#include "src/finite_difference.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <iomanip>


// Test backward difference on harder function

using Real = double;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;


Real
f(const Vector & x)
{
  // y = x*cos(1/x)
  return x(0) * std::cos(1/x(0));
}

int main()
{
  // floating point errors make i=0 not pass
  for (unsigned int i=1; i<5; ++i)
  {
    const Real pi = 2*std::acos(0);
    Vector x(1);
    x << 3/(pi+3*i*2*pi);
    Vector h(1);
    h << 1e-10;
    Vector grad_f = MEPBM::finite_difference_one_sided<Real, Vector>(&f, x, -h);
    std::cout << std::setprecision(20) << grad_f << std::endl;
  }
}