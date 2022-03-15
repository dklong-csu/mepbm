#include "src/line_search_wolfe.h"
#include <iostream>

using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;


double f(const Vector & x)
{
  return (x(0)-1)*(x(0)-1) + (x(1)-1)*(x(1)-1);
}

Vector grad_f(const Vector & x)
{
  Vector g(2);
  g(0) = 2*(x(0) - 1);
  g(1) = 2*(x(1) - 1);
  return g;
}


/*
 * This test is meant to trigger the second if statement in the line search function and immediately return alpha.
 */
int main ()
{
  Vector x(2);
  x << 0, 0;
  Vector d(2);
  d << 2, 2;
  // alpha=0.5 should result in finding the minimizer in one step


  double alpha = MEPBM::line_search_wolfe<double>(&f, &grad_f, x, d, 0.5);
  std::cout << alpha << std::endl;

}