#ifndef MEPBM_SUNDIALS_SOLVERS_H
#define MEPBM_SUNDIALS_SOLVERS_H

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "models.h"

#include <iostream>

namespace sundials
{
  /// Enumeration defining template options
  // FIXME: only DENSE is implemented right now
  enum SolverType {DENSE, SPARSEDIRECT, SPARSEITERATIVE};


  /// Enumeration to make checking of SUNDIALS C functions easier
  enum SuccessDefinition {MEMORY, RETURNZERO, RETURNNONNEGATIVE};

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
  template <SolverType Solver, typename Matrix, typename Real>
  int
  rhs_function_callback(Real t,
                        N_Vector y,
                        N_Vector y_dot,
                        void * user_data);


  /// Function to be passed to SUNDIALS to compute the Jacobian for dense matrices
  template <SolverType Solver, typename Matrix, typename Real>
  int
  setup_dense_jacobian_callback(Real t,
                                N_Vector y,
                                N_Vector y_dot,
                                SUNMatrix Jacobian,
                                void * user_data,
                                N_Vector tmp1,
                                N_Vector tmp2,
                                N_Vector tmp3);


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
        // Settings for linear solver
        const bool use_preconditioner = false,
        const bool problem_is_stiff   = false,
        // Settings for nonlinear solver
        const bool use_newton_iteration = false,
        // Integrator settings
        const Real absolute_tolerance = 1e-6,
        const Real relative_tolerance = 1e-6
        )
        : initial_time(initial_time),
          final_time(final_time),
          use_preconditioner(use_preconditioner),
          problem_is_stiff(problem_is_stiff),
          use_newton_iteration(use_newton_iteration),
          absolute_tolerance(absolute_tolerance),
          relative_tolerance(relative_tolerance)
    {}

    /// Initial time for the initial value problem
    const Real initial_time;

    /// Final time for the initial value problem
    const Real final_time;

    /// Whether or not you want to use a preconditioner in the linear solve
    const bool use_preconditioner;

    /// Whether or not the initial value problem is stiff
    const bool problem_is_stiff;

    /// Whether or not a Newton iteration is used for the nonlinear solve (fixed point iteration is used otherwise)
    const bool use_newton_iteration;

    /// Absolute tolerance for CVode
    const Real absolute_tolerance;

    /// Relative tolerance for CVode
    const Real relative_tolerance;
  };

  /// Interface to SUNDIALS CVODE solver which is suitable for stiff and nonstiff
  /// ordinary differential equations. For either case, variable-order, variable-step
  /// multistep methods are used. Adams-Moulton for nonstiff problems, and Backward Differentiation Formula
  /// for stiff problems.
  template <SolverType Solver, typename Matrix, typename Real>
  class CVodeSolver
  {
  public:
    /// Initialization parameters for CVode
    CVodeSolver(const CVodeParameters<Real> &data,
          const Model::Model<Real, Matrix> &ode_system,
          N_Vector initial_condition);

    /// Destructor
    ~CVodeSolver();

    /// Solve the system of ODEs
    void
    solve_ode(N_Vector &solution, Real tout);

    /// Solve the system of ODEs while saving the solution at intermediate points
    void
    solve_ode_incrementally(std::vector< N_Vector > &solutions,
                            const std::vector< Real > &times);

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

    /// Nonlinear solver memory structure
    SUNNonlinearSolver sun_nonlinear_solver;

    /// Vector for storing the solution
    N_Vector sun_solution_vector;

    /// Matrix template for dense solves
    SUNMatrix sun_dense_matrix;

    /// Reusable flag for checking if SUNDIALS functions executed properly
    int flag;

    /// Initial condition for the initial value problem
    const N_Vector initial_condition;

    /// Model object describing the system of ODEs
    const Model::Model< Real, Matrix > ode_system;


    /*
     * Private member functions
     */

    /// Setup the linear solver
    void
    setup_linear_solver();

    /// Setup the nonlinear solver
    void
    setup_nonlinear_solver();

    /// Setup CVode solver
    void
    setup_cvode_solver();
  };



  /***************************************************
   * SUNDIALS Dense matrix partial specialization
   **************************************************/

  template <typename Matrix, typename Real>
  class CVodeSolver<DENSE, Matrix, Real>
  {
  public:
    /// Initialization parameters for CVode
    CVodeSolver(const CVodeParameters<Real> &data,
          const Model::Model<Real, Matrix> &ode_system,
          N_Vector initial_condition);

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

    /// Nonlinear solver memory structure
    SUNNonlinearSolver sun_nonlinear_solver;

    /// Vector for storing the solution
    N_Vector sun_solution_vector;

    /// Matrix template for dense solves
    SUNMatrix sun_dense_matrix;

    /// Reusable flag for checking if SUNDIALS functions executed properly
    int flag;

    /// Initial condition for the initial value problem
    const N_Vector initial_condition;

    /// Model object describing the system of ODEs
    Model::Model< Real, Matrix > ode_system;


    /*
     * Private member functions
     */

    /// Setup the linear solver
    void
    setup_linear_solver();

    /// Setup the nonlinear solver
    void
    setup_nonlinear_solver();

    /// Setup CVode solver
    void
    setup_cvode_solver();
  };



  template <typename Matrix, typename Real>
  CVodeSolver<DENSE, Matrix, Real>::CVodeSolver(const CVodeParameters<Real> &data,
                                                const Model::Model<Real, Matrix > &ode_system,
                                                const N_Vector initial_condition)
  : data(data), ode_system(ode_system), initial_condition(initial_condition)
  {
    sun_solution_vector = N_VNew_Serial( N_VGetLength(initial_condition) );
    check_flag((void *)sun_solution_vector, "N_VNewSerial", MEMORY);

    setup_linear_solver();

    setup_nonlinear_solver();

    setup_cvode_solver();
  }


  template <typename Matrix, typename Real>
  CVodeSolver<DENSE, Matrix, Real>::~CVodeSolver()
  {
    CVodeFree(&cvode_mem);
    SUNNonlinSolFree(sun_nonlinear_solver);
    SUNLinSolFree(sun_linear_solver);
    SUNMatDestroy(sun_dense_matrix);
    N_VDestroy(sun_solution_vector);
    N_VDestroy(initial_condition);
  }


  template <typename Matrix, typename Real>
  void
  CVodeSolver<DENSE, Matrix, Real>::solve_ode(N_Vector &solution, const Real tout)
  {
    Real t;
    flag = CVode(cvode_mem, tout, solution, &t, CV_NORMAL);
    check_flag(&flag, "CVode", RETURNNONNEGATIVE);

  }


  template <typename Matrix, typename Real>
  void
  CVodeSolver<DENSE, Matrix, Real>::solve_ode_incrementally(std::vector< N_Vector > &solutions,
                                                            const std::vector< Real > &times)
  {
    N_Vector solution;
    Real t;
    // Loop through each time
    for (const auto & tout : times)
    {
      solve_ode(solution, tout);
      solutions.push_back(solution);
    }
  }



  template <typename Matrix, typename Real>
  void
  CVodeSolver<DENSE, Matrix, Real>::setup_linear_solver()
  {
    // Setup a dense linear solver
    sun_linear_solver = SUNLinSol_Dense(sun_solution_vector, sun_dense_matrix);
    check_flag( (void *)sun_linear_solver, "SUNLinSol_Dense", MEMORY);
  }


  template <typename Matrix, typename Real>
  void
  CVodeSolver<DENSE, Matrix, Real>::setup_nonlinear_solver()
  {
    if (data.use_newton_iteration)
    {
      sun_nonlinear_solver = SUNNonlinSol_Newton(sun_solution_vector);
      check_flag( (void *) sun_nonlinear_solver, "SUNNonlinSol_Newton", MEMORY);
    }
    else
    {
      sun_nonlinear_solver = SUNNonlinSol_FixedPoint(sun_solution_vector, 0);
      check_flag( (void *) sun_nonlinear_solver, "SUNNonlinSol_FixedPoint", MEMORY);
    }
  }


  template <typename Matrix, typename Real>
  void
  CVodeSolver<DENSE, Matrix, Real>::setup_cvode_solver()
  {
    // Setup integrator
    auto multistep_algorithm = (data.problem_is_stiff) ? CV_BDF : CV_ADAMS;
    cvode_mem = CVodeCreate(multistep_algorithm);
    check_flag( (void *) cvode_mem, "CVodeCreate", MEMORY);

    // Initialize integrator

    flag = CVodeInit(cvode_mem, &rhs_function_callback<DENSE, Matrix, Real>, data.initial_time, initial_condition);
    check_flag(&flag, "CVodeInit",RETURNNONNEGATIVE);

    // Specify tolerances
    flag = CVodeSStolerances(cvode_mem, data.relative_tolerance, data.absolute_tolerance);
    check_flag(&flag, "CVodeSetLinearSolver",RETURNNONNEGATIVE);

    // Attach user data
    flag = CVodeSetUserData(cvode_mem, this);
    check_flag(&flag, "CVodeSetUserData", RETURNNONNEGATIVE);

    // Attach linear solver
    flag = CVodeSetLinearSolver(cvode_mem, sun_linear_solver, sun_dense_matrix);
    check_flag(&flag, "CVodeSetLinearSolver", RETURNNONNEGATIVE);

    // Set the Jacobian routine

    flag = CVodeSetJacFn(cvode_mem, &setup_dense_jacobian_callback<DENSE, Matrix, Real>);
    check_flag(&flag, "CVodeSetJacFn", RETURNNONNEGATIVE);

    // Set stopping time
    flag = CVodeSetStopTime(cvode_mem, data.final_time);
    check_flag(&flag, "CVodeSetStopTime", RETURNNONNEGATIVE);

  }



  /// Function to be passed to SUNDIALS to compute the right-hand side
  template <SolverType Solver, typename Matrix, typename Real>
  int
  rhs_function_callback(Real t,
                        N_Vector y,
                        N_Vector y_dot,
                        void * user_data)
  {
    // user_data is a pointer to the Model describing the ODEs
    CVodeSolver<Solver, Matrix, Real> &cvode_system =
        *static_cast< CVodeSolver<Solver, Matrix, Real> *>(user_data);

    Model::Model<Real, Matrix> ode_system = cvode_system.return_ode_model();

    // Convert N_Vector to Eigen vector for compatability with Model::Model
    const auto vector_length = N_VGetLength(y);
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y_eigen(vector_length);
    auto y_data = N_VGetArrayPointer(y);
    for (unsigned int i = 0; i < vector_length; ++i)
    {
      y_eigen(i) = y_data[i];
    }

    // Calculate the right-hand side
    auto y_dot_eigen = ode_system.rhs(y_eigen);

    // Convert the Eigen vector to an N_Vector for output
    auto y_dot_data = N_VGetArrayPointer(y_dot);
    for (unsigned int i = 0; i < vector_length; ++i)
    {
      y_dot_data[i] = y_dot_eigen(i);
    }

    // Return success
    return 0;
  }


  /// Function to be passed to SUNDIALS to compute the Jacobian for dense matrices
  template <SolverType Solver, typename Matrix, typename Real>
  int
  setup_dense_jacobian_callback(Real t,
                                N_Vector y,
                                N_Vector y_dot,
                                SUNMatrix Jacobian,
                                void * user_data,
                                N_Vector tmp1,
                                N_Vector tmp2,
                                N_Vector tmp3)
  {
    // user_data is a pointer to the Model describing the ODEs
    CVodeSolver<Solver, Matrix, Real> &cvode_solver =
        *static_cast< CVodeSolver<Solver, Matrix, Real> *>(user_data);

    auto ode_system = cvode_solver.return_ode_model();

    // Convert N_Vector to Eigen vector for compatability with Model::Model
    const auto vector_length = N_VGetLength(y);
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y_eigen(vector_length);
    auto y_data = N_VGetArrayPointer(y);
    for (unsigned int i = 0; i < vector_length; ++i)
    {
      y_eigen(i) = y_data[i];
    }

    // Calculate the Jacobian
    auto Jacobian_eigen = ode_system.jacobian(y_eigen);

    // Convert the Eigen matrix into a SUNDIALS matrix
    auto Jacobian_data = SUNDenseMatrix_Data(Jacobian);
    for (unsigned int i=0; i < vector_length; ++i)
    {
      for (unsigned int j=0; j < vector_length; ++j)
      {
        // Jacobian(i,j) = data[j*M+i] where M=number of rows
        // coeffRef works with both sparse and dense Eigen matrix types
        Jacobian_data[j*vector_length + i] = Jacobian_eigen.coeffRef(i,j);
      }
    }

    // Return success
    return 0;
  }
}

#endif //MEPBM_SUNDIALS_SOLVERS_H