#include <iostream>
#include <ode_solver.h>

// Check against an ODE for which the solution is known
// x' = -10x --> x(t) = x0 * exp(-10t)
class SimpleOde
{
public:
  double step_forward(const double x, const double /*t*/, const double dt) const
  {
    // Explicit Euler for this test case
    return x + (-10 * x) * dt;
  }
};

int main()
{
  const double ic = 1.;
  const double start_time = 0.;
  const double end_time = 1.;
  const double dt = 1e-5;

  SimpleOde ode_system;

  double sol = ODE::solve_ode<SimpleOde, double>(ode_system, ic, start_time, end_time, dt);

  // The answer should be close to exp(-10)
  std::cout << sol;
}