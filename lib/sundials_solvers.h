#ifndef MEPBM_SUNDIALS_SOLVERS_H
#define MEPBM_SUNDIALS_SOLVERS_H

#include <cvode/cvode.h>
#include "nvector_eigen.h"
#include "sunmatrix_eigen.h"
#include "linear_solver_eigen.h"
#include <sunlinsol/sunlinsol_dense.h>

#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_spfgmr.h>
#include <sunlinsol/sunlinsol_spbcgs.h>
#include <sunlinsol/sunlinsol_sptfqmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "src/models.h"

#include <iostream>
#include <limits>
#include <algorithm>

namespace sundials
{
  /// Enumeration defining template options
  enum SolverType {DENSE, SPARSE};



  /// Enumeration defining iterative solvers for sparse matrices
  /// SPGMR = Scaled, Preconditioned, Generalized Minimum Residual
  /// SPFGMR = Scaled, Preconditioned, Flexible, Generalized Minimum Residual
  /// SPBCGS = Scaled, Preconditioned, BiConjugate Gradient, Stabilized
  /// SPTFQMR = Scaled, Preconditioned, Transpose-Free Quasi-Minimum Residua
  enum IterativeSolver {DIRECTSOLVE, SPGMR, SPFGMR, SPBCGS, SPTFQMR};



  /// Enumeration to make checking of SUNDIALS C functions easier
  enum SuccessDefinition {MEMORY, RETURNZERO, RETURNNONNEGATIVE};



  /// Enumeration defining nonlinear algorithm option
  enum NonlinearAlgorithm {FIXEDPOINT, NEWTON};



  /// Function to check if the C-style SUNDIALS functions are successful
  int check_flag(void *flag_value, const std::string &function_name, SuccessDefinition success_type);



  /// Function to be passed to SUNDIALS to compute the right-hand side
  template <typename Matrix, typename Real, typename SolverType>
  int
  rhs_function_callback(Real t,
                        N_Vector y,
                        N_Vector y_dot,
                        void * user_data);


  /// Function to be passed to SUNDIALS to compute the Jacobian for dense matrices
  template <typename Matrix, typename Real, typename SolverType>
  int
  setup_jacobian_callback(Real t,
                          N_Vector y,
                          N_Vector y_dot,
                          SUNMatrix Jacobian,
                          void * user_data,
                          N_Vector tmp1,
                          N_Vector tmp2,
                          N_Vector tmp3);



  /// Function to create a linear solver object
  template <typename Matrix, typename Real>
  SUNLinearSolver
  setup_linear_solver(IterativeSolver solver_type,
                      N_Vector v,
                      SUNMatrix M);



  SUNNonlinearSolver
  create_nonlinear_solver(NonlinearAlgorithm algorithm,
                          N_Vector v);



  /// Initialization parameters for CVode
  template <typename Real>
  class CVodeParameters
  {
  public:
    /// Constructor
    explicit CVodeParameters(
        // Initial parameters
        const Real initial_time = 0.0,
        const Real final_time = 1.0,
        // Integrator settings
        const Real absolute_tolerance = 1e-6,
        const Real relative_tolerance = 1e-6,
        const int solver_type = CV_ADAMS
        )
        : initial_time(initial_time),
          final_time(final_time),
          absolute_tolerance(absolute_tolerance),
          relative_tolerance(relative_tolerance),
          solver_type(solver_type)
    {}

    /// Initial time for the initial value problem
    const Real initial_time;

    /// Final time for the initial value problem
    const Real final_time;

    /// Absolute tolerance for CVode
    const Real absolute_tolerance;

    /// Relative tolerance for CVode
    const Real relative_tolerance;

    /// ODE solver algorithm. CV_ADAMS == 1 (for nonstiff only), CV_BDF == 2
    const int solver_type;
  };

  /// Interface to SUNDIALS CVODE solver which is suitable for stiff and nonstiff
  /// ordinary differential equations. For either case, variable-order, variable-step
  /// multistep methods are used. Adams-Moulton for nonstiff problems, and Backward Differentiation Formula
  /// for stiff problems.
  template <typename Matrix, typename Real, typename SolverType>
  class CVodeSolver
  {
  public:
    /// Initialization parameters for CVode
    CVodeSolver(const CVodeParameters<Real> &data,
                const Model::Model<Real, Matrix> &ode_system,
                N_Vector initial_condition,
                N_Vector sun_solution_vector,
                SUNMatrix sun_matrix,
                SUNLinearSolver linear_solver);

    /// Destructor
    ~CVodeSolver();

    /// Solve the system of ODEs
    int
    solve_ode(N_Vector &solution, Real tout);

    /// Solve the system of ODEs while saving the solution at intermediate points
    int
    solve_ode_incrementally(std::vector< N_Vector > &solutions,
                            const std::vector< Real > &times);

    /// Returns the model describing the ODEs
    Model::Model< Real, Matrix>
    return_ode_model();

    /// Returns the CVodeParameters
    CVodeParameters<Real>
    return_cvode_settings();

  private:
    /*
     * Member variables
     */

    /// Data for setting up the ODE solver
    const CVodeParameters<Real> data;

    /// Memory for solving the ODE
    void *cvode_mem;

    /// Linear solver memory structure
    SUNLinearSolver sun_linear_solver;

    /// Vector for storing the solution
    N_Vector sun_solution_vector;

    /// Matrix template for dense solves
    SUNMatrix sun_matrix;

    /// Reusable flag for checking if SUNDIALS functions executed properly
    int flag;

    /// Initial condition for the initial value problem
    N_Vector initial_condition;

    /// Model object describing the system of ODEs
    const Model::Model< Real, Matrix > ode_system;


    /*
     * Private member functions
     */

    /// Setup CVode solver
    void
    setup_cvode_solver();
  };



  /***************************************************
   * Implementation of CVodeSolver
   **************************************************/

  template <typename Matrix, typename Real, typename SolverType>
  CVodeSolver<Matrix, Real, SolverType>::CVodeSolver(const CVodeParameters<Real> &data,
                                                     const Model::Model<Real, Matrix > &ode_system,
                                                     const N_Vector ic,
                                                     const N_Vector nvector_template,
                                                     const SUNMatrix sunmatrix_template,
                                                     const SUNLinearSolver linear_solver)
  : data(data), ode_system(ode_system),
    initial_condition(nullptr),
    sun_linear_solver(nullptr),
    cvode_mem(nullptr),
    sun_solution_vector(nullptr),
    sun_matrix(nullptr),
    flag(-1)
  {
    // Clone the initial condition argument and fill the vector
    initial_condition = ic->ops->nvclone(ic);
    auto new_ic_ptr = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(initial_condition->content);
    auto old_ic_ptr = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(ic->content);
    assert(new_ic_ptr->size() == old_ic_ptr->size());
    for (unsigned int i=0;i<new_ic_ptr->size();++i)
    {
      (*new_ic_ptr)(i) = (*old_ic_ptr)(i);
    }

    // Clone the vector template -- SUNDIALS is fine with the underlying vector being uninitialized.
    sun_solution_vector = nvector_template->ops->nvclone(nvector_template);

    // Clone the matrix template -- SUNDIALS is fine with the underlying matrix being uninitialized.
    sun_matrix = sunmatrix_template->ops->clone(sunmatrix_template);

    // Clone the linear solver
    if (linear_solver->ops->gettype(linear_solver) == SUNLINEARSOLVER_DIRECT)
      sun_linear_solver = create_eigen_direct_linear_solver<Matrix, Real, SolverType>();
    else
      sun_linear_solver = create_eigen_iterative_linear_solver<Matrix, Real, SolverType>();

    sun_linear_solver->ops->gettype = linear_solver->ops->gettype;
    sun_linear_solver->ops->setup = linear_solver->ops->setup;
    sun_linear_solver->ops->solve = linear_solver->ops->solve;
    sun_linear_solver->ops->free = linear_solver->ops->free;


    setup_cvode_solver();
  }


  template <typename Matrix, typename Real, typename SolverType>
  CVodeSolver<Matrix, Real, SolverType>::~CVodeSolver()
  {
    CVodeFree(&cvode_mem);
    sun_linear_solver->ops->free(sun_linear_solver);
    sun_matrix->ops->destroy(sun_matrix);
    sun_solution_vector->ops->nvdestroy(sun_solution_vector);
    initial_condition->ops->nvdestroy(initial_condition);
  }


  template <typename Matrix, typename Real, typename SolverType>
  int
  CVodeSolver<Matrix, Real, SolverType>::solve_ode(N_Vector &solution, const Real tout)
  {
    Real t;
    flag = CVode(cvode_mem, tout, solution, &t, CV_NORMAL);
    check_flag(&flag, "CVode", RETURNNONNEGATIVE);
    return flag;
  }


  template <typename Matrix, typename Real, typename SolverType>
  int
  CVodeSolver<Matrix, Real, SolverType>::solve_ode_incrementally(std::vector< N_Vector > &solutions,
                                                     const std::vector< Real > &times)
  {
    Real t;
    int err_flag;
    // Loop through each time
    for (const auto & tout : times)
    {
      N_Vector solution = N_VClone(sun_solution_vector);
      auto sol_vec_ptr = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(solution->content);
      for (unsigned int i=0; i<sol_vec_ptr->size();++i)
      {
        (*sol_vec_ptr)(i) = 0.;
      }
      err_flag = solve_ode(solution, tout);
      if (err_flag < 0)
      {
        // This means the solver failed for some reason and we don't care about the solutions
        return err_flag;
      }
      solutions.push_back(solution);
    }
    return err_flag;
  }



  template <typename Matrix, typename Real, typename SolverType>
  void
  CVodeSolver<Matrix, Real, SolverType>::setup_cvode_solver()
  {
    // Setup integrator
    cvode_mem = CVodeCreate(data.solver_type);
    check_flag( (void *) cvode_mem, "CVodeCreate", MEMORY);


    // Initialize integrator
    flag = CVodeInit(cvode_mem, &rhs_function_callback<Matrix, Real, SolverType>, data.initial_time, initial_condition);
    check_flag(&flag, "CVodeInit",RETURNNONNEGATIVE);


    // Specify tolerances
    flag = CVodeSStolerances(cvode_mem, data.relative_tolerance, data.absolute_tolerance);
    check_flag(&flag, "CVodeSStolerances",RETURNNONNEGATIVE);


    // Set maximum number of steps -- a negative value disables the test, which is desired
    flag = CVodeSetMaxNumSteps(cvode_mem, 5000);
    check_flag(&flag, "CVodeSetMaxNumSteps", RETURNNONNEGATIVE);


    // Attach user data
    flag = CVodeSetUserData(cvode_mem, this);
    check_flag(&flag, "CVodeSetUserData", RETURNNONNEGATIVE);


    // Attach linear solver
    flag = CVodeSetLinearSolver(cvode_mem, sun_linear_solver, sun_matrix);
    check_flag(&flag, "CVodeSetLinearSolver", RETURNNONNEGATIVE);


    // Set the Jacobian routine
    flag = CVodeSetJacFn(cvode_mem, &setup_jacobian_callback<Matrix, Real, SolverType>);
    check_flag(&flag, "CVodeSetJacFn", RETURNNONNEGATIVE);


    // Set stopping time
    flag = CVodeSetStopTime(cvode_mem, data.final_time);
    check_flag(&flag, "CVodeSetStopTime", RETURNNONNEGATIVE);
  }



  /// Returns the ODE model
  template< typename Matrix, typename Real, typename SolverType>
  Model::Model< Real, Matrix>
  CVodeSolver<Matrix, Real, SolverType>::return_ode_model()
  {
    return ode_system;
  }



  /// Returns the CVode parameters
  template< typename Matrix, typename Real, typename SolverType>
  CVodeParameters< Real>
  CVodeSolver<Matrix, Real, SolverType>::return_cvode_settings()
  {
    return data;
  }



  /******************
   * HELPER FUNCTIONS
   *****************/

  /// Function to check if the C-style SUNDIALS functions are successful
  int check_flag(void *flag_value, const std::string &function_name, SuccessDefinition success_type)
  {
    int result = 0;

    // For checking when success is defined by a return value.
    int error_flag;

    switch (success_type) {
      case MEMORY :
        if (flag_value == nullptr)
        {
          std::cerr << std::endl
                    << "ERROR: "
                    << function_name
                    << " returned NULL pointer."
                    << std::endl;
          // Return 1 to indicate failure
          result = 1;
        }
        break;

      case RETURNZERO :
        error_flag = *((int *) flag_value);
        if (error_flag != 0)
        {
          std::cerr << std::endl
                    << "ERROR: "
                    << function_name
                    << " returned with flag = "
                    << error_flag
                    << std::endl;
          // Return 1 to indicate failure
          result = 1;
        }
        break;

      case RETURNNONNEGATIVE :
        error_flag = *((int *) flag_value);
        if (error_flag < 0)
        {
          std::cerr << std::endl
                    << "ERROR: "
                    << function_name
                    << " returned with flag = "
                    << error_flag
                    << std::endl;
          // Return 1 to indicate failure
          result = 1;
        }
        break;

      default:
        std::cerr << std::endl
                  << "ERROR: check_flag called with an invalid SuccessDefinition."
                  << std::endl;
        // Return 1 to indicate failure
        result = 1;
        break;
    }

    return result;
  }



  /// Function to be passed to SUNDIALS to compute the right-hand side
  template <typename Matrix, typename Real, typename SolverType>
  int
  rhs_function_callback(Real t,
                        N_Vector y,
                        N_Vector y_dot,
                        void * user_data)
  {
    // user_data is a pointer to the Model describing the ODEs
    CVodeSolver<Matrix, Real, SolverType> &cvode_system =
        *static_cast< CVodeSolver<Matrix, Real, SolverType> *>(user_data);

    Model::Model<Real, Matrix> ode_system = cvode_system.return_ode_model();

    auto y_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1> *>(y->content);
    auto y_dot_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1> *>(y_dot->content);

    *y_dot_vec = ode_system.rhs(*y_vec);

    // Return success
    return 0;
  }


  /// Function to be passed to SUNDIALS to compute the Jacobian for dense matrices
  template <typename Matrix, typename Real, typename SolverType>
  int
  setup_jacobian_callback(Real t,
                          N_Vector y,
                          N_Vector y_dot,
                          SUNMatrix Jacobian,
                          void * user_data,
                          N_Vector tmp1,
                          N_Vector tmp2,
                          N_Vector tmp3)
  {
    // user_data is a pointer to the Model describing the ODEs
    CVodeSolver<Matrix, Real, SolverType> &cvode_solver =
        *static_cast< CVodeSolver<Matrix, Real, SolverType> *>(user_data);

    const auto cvode_settings = cvode_solver.return_cvode_settings();

    const auto ode_system = cvode_solver.return_ode_model();

    auto J_mat = static_cast<Matrix*>(Jacobian->content);
    auto y_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1>* >(y->content);

    *J_mat = ode_system.jacobian(*y_vec);

    // Return success
    return 0;
  }
}

#endif //MEPBM_SUNDIALS_SOLVERS_H
