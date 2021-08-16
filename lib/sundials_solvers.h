#ifndef MEPBM_SUNDIALS_SOLVERS_H
#define MEPBM_SUNDIALS_SOLVERS_H

#include <cvode/cvode.h>
#include "nvector_eigen.h"
#include "sunmatrix_eigen.h"
#include <sunlinsol/sunlinsol_dense.h>

#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_spfgmr.h>
#include <sunlinsol/sunlinsol_spbcgs.h>
#include <sunlinsol/sunlinsol_sptfqmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "models.h"

#include <iostream>
#include <limits>

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
  template <typename Matrix, typename Real>
  int
  rhs_function_callback(Real t,
                        N_Vector y,
                        N_Vector y_dot,
                        void * user_data);


  /// Function to be passed to SUNDIALS to compute the Jacobian for dense matrices
  template <typename Matrix, typename Real>
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
  template <typename Matrix, typename Real>
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
    void
    solve_ode(N_Vector &solution, Real tout);

    /// Solve the system of ODEs while saving the solution at intermediate points
    void
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
    const N_Vector initial_condition;

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

  template <typename Matrix, typename Real>
  CVodeSolver<Matrix, Real>::CVodeSolver(const CVodeParameters<Real> &data,
                                         const Model::Model<Real, Matrix > &ode_system,
                                         const N_Vector initial_condition,
                                         const N_Vector sun_solution_vector,
                                         const SUNMatrix sun_matrix,
                                         const SUNLinearSolver linear_solver)
  : data(data), ode_system(ode_system),
    initial_condition(initial_condition),
    sun_linear_solver(linear_solver),
    cvode_mem(nullptr),
    sun_solution_vector(sun_solution_vector),
    sun_matrix(sun_matrix),
    flag(-1)
  {
    setup_cvode_solver();
  }


  template <typename Matrix, typename Real>
  CVodeSolver<Matrix, Real>::~CVodeSolver()
  {
    CVodeFree(&cvode_mem);
    SUNLinSolFree(sun_linear_solver);
    SUNMatDestroy(sun_matrix);
    N_VDestroy(sun_solution_vector);
    N_VDestroy(initial_condition);
  }


  template <typename Matrix, typename Real>
  void
  CVodeSolver<Matrix, Real>::solve_ode(N_Vector &solution, const Real tout)
  {
    Real t;
    flag = CVode(cvode_mem, tout, solution, &t, CV_NORMAL);
    check_flag(&flag, "CVode", RETURNNONNEGATIVE);

  }


  template <typename Matrix, typename Real>
  void
  CVodeSolver<Matrix, Real>::solve_ode_incrementally(std::vector< N_Vector > &solutions,
                                                     const std::vector< Real > &times)
  {
    Real t;
    // Loop through each time
    for (const auto & tout : times)
    {
      N_Vector solution = N_VClone(sun_solution_vector);
      solve_ode(solution, tout);
      solutions.push_back(solution);
    }
  }



  template <typename Matrix, typename Real>
  void
  CVodeSolver<Matrix, Real>::setup_cvode_solver()
  {
    // Setup integrator
    cvode_mem = CVodeCreate(data.solver_type);
    check_flag( (void *) cvode_mem, "CVodeCreate", MEMORY);


    // Initialize integrator
    flag = CVodeInit(cvode_mem, &rhs_function_callback<Matrix, Real>, data.initial_time, initial_condition);
    check_flag(&flag, "CVodeInit",RETURNNONNEGATIVE);


    // Specify tolerances
    flag = CVodeSStolerances(cvode_mem, data.relative_tolerance, data.absolute_tolerance);
    check_flag(&flag, "CVodeSStolerances",RETURNNONNEGATIVE);


    // Set maximum number of steps
    flag = CVodeSetMaxNumSteps(cvode_mem, initial_condition->ops->nvgetlength(initial_condition)*10);
    check_flag(&flag, "CVodeSetMaxNumSteps", RETURNNONNEGATIVE);


    // Attach user data
    flag = CVodeSetUserData(cvode_mem, this);
    check_flag(&flag, "CVodeSetUserData", RETURNNONNEGATIVE);


    // Attach linear solver
    flag = CVodeSetLinearSolver(cvode_mem, sun_linear_solver, sun_matrix);
    check_flag(&flag, "CVodeSetLinearSolver", RETURNNONNEGATIVE);


    // Set the Jacobian routine
    flag = CVodeSetJacFn(cvode_mem, &setup_jacobian_callback<Matrix, Real>);
    check_flag(&flag, "CVodeSetJacFn", RETURNNONNEGATIVE);


    // Set stopping time
    flag = CVodeSetStopTime(cvode_mem, data.final_time);
    check_flag(&flag, "CVodeSetStopTime", RETURNNONNEGATIVE);
  }



  /// Returns the ODE model
  template< typename Matrix, typename Real >
  Model::Model< Real, Matrix>
  CVodeSolver<Matrix, Real >::return_ode_model()
  {
    return ode_system;
  }



  /// Returns the CVode parameters
  template< typename Matrix, typename Real >
  CVodeParameters< Real>
  CVodeSolver<Matrix, Real >::return_cvode_settings()
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
  template <typename Matrix, typename Real>
  int
  rhs_function_callback(Real t,
                        N_Vector y,
                        N_Vector y_dot,
                        void * user_data)
  {
    // user_data is a pointer to the Model describing the ODEs
    CVodeSolver<Matrix, Real> &cvode_system =
        *static_cast< CVodeSolver<Matrix, Real> *>(user_data);

    Model::Model<Real, Matrix> ode_system = cvode_system.return_ode_model();

    auto y_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1> *>(y->content);
    auto y_dot_vec = static_cast< Eigen::Matrix<Real, Eigen::Dynamic, 1> *>(y_dot->content);

    *y_dot_vec = ode_system.rhs(*y_vec);

    // Return success
    return 0;
  }


  /// Function to be passed to SUNDIALS to compute the Jacobian for dense matrices
  template <typename Matrix, typename Real>
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
    CVodeSolver<Matrix, Real> &cvode_solver =
        *static_cast< CVodeSolver<Matrix, Real> *>(user_data);

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
