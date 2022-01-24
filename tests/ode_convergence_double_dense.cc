#include <iostream>
#include <iomanip>
#include "src/ode_solver.h"
#include <vector>
#include <cmath>
#include "src/models.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>



using Real = double;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;



class SimpleOde : public Model::RightHandSideContribution<Real, Matrix>
{
  void add_contribution_to_rhs(const Vector &x, Vector &rhs)
  {
    rhs += -10 * x;
  }

  void add_contribution_to_jacobian(const Vector &x, Matrix &jacobi)
  {
    jacobi += -10 * Matrix::Identity(x.rows(), x.rows());
  }

  void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) {}

  void update_num_nonzero(unsigned int &num_nonzero) {}
};



int main()
{
  std::vector<Real> dt_vals = { 1./2 , 1./4, 1./8, 1./16, 1./32, 1./64, 1./128, 1./256, 1./512, 1./1024};

  const int digits = 100;

  std::cout << "dt = [";
  for (auto val : dt_vals)
  {
    std::cout << std::setprecision(digits) << std::fixed << val << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;


  std::cout << "bdf1 = [";
  for (auto val : dt_vals)
  {
    const Vector ic = Vector::Ones(1);
    const Real start_time = 0.;
    const Real end_time = 1.;
    const Real dt = val;

    std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model<Real, Matrix> ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperBDF<1, Real, Matrix> stepper(ode_system);
    auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const Real sol_double = sol(0);
    const Real exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "bdf2 = [";
  for (auto val : dt_vals)
  {
    const Vector ic = Vector::Ones(1);
    const Real start_time = 0.;
    const Real end_time = 1.;
    const Real dt = val;

    std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model<Real, Matrix> ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperBDF<2, Real, Matrix> stepper(ode_system);
    auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const Real sol_double = sol(0);
    const Real exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "bdf3 = [";
  for (auto val : dt_vals)
  {
    const Vector ic = Vector::Ones(1);
    const Real start_time = 0.;
    const Real end_time = 1.;
    const Real dt = val;

    std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model<Real, Matrix> ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperBDF<3, Real, Matrix> stepper(ode_system);
    auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const Real sol_double = sol(0);
    const Real exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "bdf4 = [";
  for (auto val : dt_vals)
  {
    const Vector ic = Vector::Ones(1);
    const Real start_time = 0.;
    const Real end_time = 1.;
    const Real dt = val;

    std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model<Real, Matrix> ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperBDF<4, Real, Matrix> stepper(ode_system);
    auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const Real sol_double = sol(0);
    const Real exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "sdirk1 = [";
  for (auto val : dt_vals)
  {
    const Vector ic = Vector::Ones(1);
    const Real start_time = 0.;
    const Real end_time = 1.;
    const Real dt = val;

    std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model<Real, Matrix> ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperSDIRK<1, Real, Matrix> stepper(ode_system);
    auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const Real sol_double = sol(0);
    const Real exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "sdirk2 = [";
  for (auto val : dt_vals)
  {
    const Vector ic = Vector::Ones(1);
    const Real start_time = 0.;
    const Real end_time = 1.;
    const Real dt = val;

    std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model<Real, Matrix> ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperSDIRK<2, Real, Matrix> stepper(ode_system);
    auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const Real sol_double = sol(0);
    const Real exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "sdirk3 = [";
  for (auto val : dt_vals)
  {
    const Vector ic = Vector::Ones(1);
    const Real start_time = 0.;
    const Real end_time = 1.;
    const Real dt = val;

    std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model<Real, Matrix> ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperSDIRK<3, Real, Matrix> stepper(ode_system);
    auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const Real sol_double = sol(0);
    const Real exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl << std::endl;



  std::cout << "sdirk3 = [";
  for (auto val : dt_vals)
  {
    const Vector ic = Vector::Ones(1);
    const Real start_time = 0.;
    const Real end_time = 1.;
    const Real dt = val;

    std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
        = std::make_shared<SimpleOde>();
    Model::Model<Real, Matrix> ode_system(0, 0);
    ode_system.add_rhs_contribution(my_ode);
    ODE::StepperSDIRK<4, Real, Matrix> stepper(ode_system);
    auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

    // The answer should be close to exp(-10)
    const Real sol_double = sol(0);
    const Real exact_sol = std::exp(-10);
    auto abs_diff = std::abs(sol_double - exact_sol);
    std::cout << std::setprecision(digits) << std::fixed << abs_diff << std::endl;
  }
  std::cout << "];" << std::endl;
}
