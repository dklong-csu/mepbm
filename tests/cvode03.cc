#include "src/cvode.h"
#include "src/create_sunlinearsolver.h"
#include "src/create_nvector.h"
#include "src/create_sunmatrix.h"
#include "src/chemical_reaction.h"
#include "eigen3/Eigen/Dense"
#include <iostream>
#include <iomanip>


using Real = realtype;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using Solver = Eigen::ColPivHouseholderQR< Matrix >;


/*
 * Make the ODE system be
 *    dx/dt = -10*x
 *    x(0) = 1
 * so that we know the solution is x_i = e^(-10t)
 */
int rhs(Real t, N_Vector x, N_Vector x_dot, void * user_data)
{
  const Real factor = *(Real *)user_data;
  Vector* x_vec = static_cast<Vector*>(x->content);
  Vector* x_dot_vec = static_cast<Vector*>(x_dot->content);

  (*x_dot_vec) = factor * (*x_vec);
  return 0; // success!
}



/*
 * Based on the above ODE, the Jacobian is J=-10
 */
int jac(Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void * user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  const Real factor = *(Real *)user_data;
  J->ops->zero(J);
  Matrix* J_mat = static_cast<Matrix*>(J->content);
  auto dim = J_mat->rows();
  for (unsigned int i=0; i<dim; ++i)
  {
    (*J_mat)(i,i) = factor;
  }
  return 0; // success!
}



int main ()
{
  // Create initial condition
  auto ic = MEPBM::create_eigen_nvector<Vector>(1);
  Vector* ic_vec = static_cast<Vector*>(ic->content);
  (*ic_vec) = Vector::Ones(1);

  // State start and end times
  const Real t0 = 0;
  const Real t1 = 1;

  // Create the matrix template
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(1,1);

  // Create the linear solver
  auto linear_solver = MEPBM::create_dense_direct_solver<Matrix, Real, Solver>();

  // Create the CVODE object
  MEPBM::CVODE<Real> ode_solver(ic,template_matrix, linear_solver,&rhs,&jac,t0,t1);

  // Provide the exponent for the intended solution
  const Real factor = -10;
  auto factor_ptr = &factor;
  auto user_data = (void *)factor_ptr;
  ode_solver.set_user_data(user_data);

  // Solve the ODE
  auto solution = ode_solver.solve(t1);

  // The answer should be close to exp(-10)
  const Vector s = *static_cast<Vector*>(solution->content);
  std::cout << std::setprecision(20) << std::fixed << s << std::endl;
}