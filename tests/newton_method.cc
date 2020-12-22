#include <iostream>
#include "ode_solver.h"
#include <Eigen/Dense>

using namespace Eigen;

class Function
{
public:
  Function(const Vector2d x)
    : first_guess(x)
  {
    Matrix2d J;
    J(0,0) = 2*x(0);
    J(0,1) = 0.;
    J(1,0) = 0.;
    J(1,1) = 2*x(1);

    jacobian = J.partialPivLu();
  }

  Vector2d value(Vector2d &x) const
  {
    Vector2d sol;
    sol << x(0)*x(0), x(1)*x(1);
    return sol;
  }

  const Vector2d first_guess;
  PartialPivLU<Matrix2d> jacobian;
};


int main()
{
  Vector2d guess;
  guess << 2, 1;
  Function my_function(guess);

  auto sol = ODE::perform_newtons_method<Function, Vector2d>(my_function, guess, 1e-10, 100);

  std::cout << sol;
}