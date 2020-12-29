#include <iostream>
#include "ode_solver.h"
#include <eigen3/Eigen/Dense>



class Fcn : public ODE::FunctionBase
{
public:
  Eigen::VectorXd value(const Eigen::VectorXd &x) const override;
};


Eigen::VectorXd Fcn::value(const Eigen::VectorXd &x) const
{
  Eigen::VectorXd sol(2);
  sol(0) = x(0) * x(0);
  sol(1) = x(1) * x(1);
  return sol;
}



int main()
{
  Eigen::VectorXd guess(2);
  guess(0) = 2;
  guess(1) = 1;

  Eigen::MatrixXd J(2,2);
  J(0, 0) = 4;
  J(0, 1) = 0;
  J(1, 0) = 0;
  J(1, 1) = 2;

  auto J_inverse = J.partialPivLu();

  Fcn fcn;

  auto newton_result = ODE::newton_method(fcn, J_inverse, guess);

  // The answer should be close to [0, 0].
  std::cout << newton_result.first;
}