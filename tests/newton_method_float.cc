#include <iostream>
#include "src/ode_solver.h"
#include <eigen3/Eigen/Dense>



using Real = float;



class Fcn : public ODE::FunctionBase<Real>
{
public:
  Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;
};


Eigen::Matrix<Real, Eigen::Dynamic, 1> Fcn::value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
{
  Eigen::Matrix<Real, Eigen::Dynamic, 1> sol(2);
  sol(0) = x(0) * x(0);
  sol(1) = x(1) * x(1);
  return sol;
}



int main()
{
  Eigen::Matrix<Real, Eigen::Dynamic, 1> guess(2);
  guess(0) = 3/4;
  guess(1) = 1/2;

  Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> J(2,2);
  J(0, 0) = 3/2;
  J(0, 1) = 0;
  J(1, 0) = 0;
  J(1, 1) = 1;

  auto J_inverse = J.partialPivLu();

  Fcn fcn;

  auto newton_result = ODE::newton_method<Real>(fcn, J_inverse, guess, 1e-10, 1000);

  // The answer should be close to [0, 0].
  std::cout << newton_result.first;
}