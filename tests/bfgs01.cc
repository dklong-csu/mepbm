#include <eigen3/Eigen/Dense>
#include "src/bfgs.h"
#include <iostream>

using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;


double f(const Vector & x)
{
  return (x(0)-1)*(x(0)-1) +(x(1)-1)*(x(1)-1);
}

Vector grad_f(const Vector & x)
{
  Vector g(2);
  g << 2*(x(0)-1), 2*(x(1)-1);
  return g;
}


/*
 * This test tries to minimize f(x,y) = (x-1)^2 + (y-1)^2
 * which has a unique minimizer at [1,1]
 */
int main ()
{
  MEPBM::BFGS<double> bfgs(&f, &grad_f);
  Vector x(2);
  x << 4, -1;
  Vector x_star = bfgs.minimize(x, 1e-6, 100);
  std::cout << x_star << std::endl;
}