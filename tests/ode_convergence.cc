#include <iostream>
#include <iomanip>
#include <ode_solver.h>
#include <vector>
#include <cmath>
#include "models.h"
#include <eigen3/Eigen/Dense>



class SimpleOde : public Model::RightHandSideContribution
{
  void add_contribution_to_rhs(const Eigen::VectorXd &x, Eigen::VectorXd &rhs)
  {
    rhs += -10 * x;
  }

  void add_contribution_to_jacobian(const Eigen::VectorXd &x, Eigen::MatrixXd &jacobi)
  {
    jacobi += -10 * Eigen::MatrixXd::Identity(x.rows(), x.rows());
  }
};

int main()
{
  std::vector<double> dt_vals = { 1./2 , 1./4, 1./8, 1./16, 1./32, 1./64, 1./128, 1./256};


  std::cout << "dt = [";
  std::cout << dt_vals[0];
  for (unsigned int i=1; i < dt_vals.size(); ++i)
  {
    std::cout << "; " << dt_vals[i];
  }
  std::cout << "];" << std::endl << std::endl;


  const int digits = 100;

  std::cout << "bdf1 = [";
  for (auto val : dt_vals)
  {
    const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
    const double start_time = 0.;
    const double end_time = 1.;
    const double dt = val;

    std::shared_ptr<Model::RightHandSideContribution> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperBDF<1> stepper(ode_system);
    auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const double sol_double = sol(0);
    const double exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "bdf2 = [";
  for (auto val : dt_vals)
  {
    const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
    const double start_time = 0.;
    const double end_time = 1.;
    const double dt = val;

    std::shared_ptr<Model::RightHandSideContribution> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperBDF<2> stepper(ode_system);
    auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const double sol_double = sol(0);
    const double exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "bdf3 = [";
  for (auto val : dt_vals)
  {
    const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
    const double start_time = 0.;
    const double end_time = 1.;
    const double dt = val;

    std::shared_ptr<Model::RightHandSideContribution> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperBDF<3> stepper(ode_system);
    auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const double sol_double = sol(0);
    const double exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "bdf4 = [";
  for (auto val : dt_vals)
  {
    const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
    const double start_time = 0.;
    const double end_time = 1.;
    const double dt = val;

    std::shared_ptr<Model::RightHandSideContribution> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperBDF<4> stepper(ode_system);
    auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const double sol_double = sol(0);
    const double exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "sdirk1 = [";
  for (auto val : dt_vals)
  {
    const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
    const double start_time = 0.;
    const double end_time = 1.;
    const double dt = val;

    std::shared_ptr<Model::RightHandSideContribution> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperSDIRK<1> stepper(ode_system);
    auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const double sol_double = sol(0);
    const double exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "sdirk2 = [";
  for (auto val : dt_vals)
  {
    const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
    const double start_time = 0.;
    const double end_time = 1.;
    const double dt = val;

    std::shared_ptr<Model::RightHandSideContribution> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperSDIRK<2> stepper(ode_system);
    auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const double sol_double = sol(0);
    const double exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "sdirk3 = [";
  for (auto val : dt_vals)
  {
    const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
    const double start_time = 0.;
    const double end_time = 1.;
    const double dt = val;

    std::shared_ptr<Model::RightHandSideContribution> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperSDIRK<3> stepper(ode_system);
    auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const double sol_double = sol(0);
    const double exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "sdirk3 = [";
  for (auto val : dt_vals)
  {
    const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
    const double start_time = 0.;
    const double end_time = 1.;
    const double dt = val;

    std::shared_ptr<Model::RightHandSideContribution> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperSDIRK<4> stepper(ode_system);
    auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const double sol_double = sol(0);
    const double exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl;
}
