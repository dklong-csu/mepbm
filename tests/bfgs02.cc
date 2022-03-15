#include <eigen3/Eigen/Dense>
#include "src/bfgs.h"
#include <iostream>
#include <cmath>

using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;


double f(const Vector & v)
{
  const double x = v(0);
  const double y = v(1);
  const double a = 1;
  const double b = 100;
  return (a-x)*(a-x) + b*(y-x*x)*(y-x*x);
}

Vector grad_f(const Vector & v)
{
  const double x = v(0);
  const double y = v(1);
  const double a = 1;
  const double b = 100;

  Vector g(2);
  g(0) = 2*(x-a) - 4*b*x*(y-x*x);
  g(1) = 2*b*(y-x*x);
  return g;
}


/*
 * This test tries to minimize f(x,y) = (1-x)^2+100(y-x^2)^2, also known as the Rosenbrock function,
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