#ifndef MEPBM_CVODE_H
#define MEPBM_CVODE_H

#include <cvode/cvode.h>
#include "create_nvector.h"
#include "create_sunmatrix.h"
#include "create_sunlinearsolver.h"
#include "check_sundials_flags.h"

#include <sunnonlinsol/sunnonlinsol_newton.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <functional>



namespace MEPBM {
  /**
   * Object encapsulating the SUNDIALS CVODE solver. Set up to automate all of the setup steps so that the user
   * just needs to tell the object what time to solve for. The underlying object SUNDIALS uses can be modified
   * away from the default settings through the provided functions.
   */
   template <typename Real>
   class CVODE
   {
   public:
     /// Constructor
     CVODE(N_Vector initial_condition,
           SUNMatrix template_matrix,
           SUNLinearSolver linear_solver,
           /*This just means to pass a function pointer that returns an `int` and has `Real, N_Vector, N_Vector, void*` as parameters*/
           int (*rhs_function)(Real, N_Vector, N_Vector, void*),
           /*This just means to pass a function pointer that returns an `int` and has `Real, N_Vector, N_Vector, SUNMatrix, void *, N_Vector, N_Vector, N_Vector` as parameters*/
           int (*jacobian_function)(Real, N_Vector, N_Vector, SUNMatrix, void *, N_Vector, N_Vector, N_Vector),
           Real initial_time,
           Real final_time)
     : template_vector(nullptr),
       cvode_memory(nullptr),
       flag(-1),
       initial_time(initial_time),
       final_time(final_time)
     {
       // Used for checking when SUNDIALS functions are called.
       int check = 0;


       // Use the provided initial condition as a template vector
       template_vector = initial_condition->ops->nvclone(initial_condition);


       // For the problems this code is meant to solve, the ODEs are almost always stiff.
       // So we want to use BDF solvers instead of Adams solvers.
       cvode_memory = CVodeCreate(CV_BDF);
       check = check_flag( (void *) cvode_memory, "CVodeCreate", MEMORY);
       assert(check == 0);


       // Initialize CVODE
       flag = CVodeInit(cvode_memory, rhs_function, initial_time, initial_condition);
       check = check_flag(&flag, "CVodeInit",RETURNNONNEGATIVE);
       assert(check == 0);


       // Specify tolerances -- these can be modified with the `set_tolerance()` function prior to calling `solve()`
       // Based on my experience with the equations that arise in MEPBM applications, these default tolerances are pretty good.
       flag = CVodeSStolerances(cvode_memory, 1e-6, 1e-12);
       check = check_flag(&flag, "CVodeSStolerances",RETURNNONNEGATIVE);
       assert(check == 0);


       // Set maximum number of steps -- a negative value disables the test which is what we want by default
       // This really should not be modified because if the solver fails to reach the desired time then the
       // ODE solution cannot be trusted and should not be used for subsequent calculations.
       flag = CVodeSetMaxNumSteps(cvode_memory, -1);
       check = check_flag(&flag, "CVodeSetMaxNumSteps", RETURNNONNEGATIVE);
       assert(check == 0);


       // Attach linear solver
       flag = CVodeSetLinearSolver(cvode_memory, linear_solver, template_matrix);
       check = check_flag(&flag, "CVodeSetLinearSolver", RETURNNONNEGATIVE);
       assert(check == 0);


       // Set the Jacobian routine
       flag = CVodeSetJacFn(cvode_memory, jacobian_function);
       check = check_flag(&flag, "CVodeSetJacFn", RETURNNONNEGATIVE);
       assert(check == 0);


       // Set stopping time
       flag = CVodeSetStopTime(cvode_memory, final_time);
       check = check_flag(&flag, "CVodeSetStopTime", RETURNNONNEGATIVE);
       assert(check == 0);
     }


     /// Destructor
     ~CVODE()
     {
       // All the SUNDIALS data structures are created on the Heap, so we need to dispose of them here.
       template_vector->ops->nvdestroy(template_vector);
       CVodeFree(&cvode_memory);
     }

     /// Function to set the absolute and relative tolerances. By default abs_tol=1e-12 and rel_tol=1e-6.
     void
     set_tolerance(const Real rel_tol, const Real abs_tol)
     {
       int check = 0;
       flag = CVodeSStolerances(cvode_memory, rel_tol, abs_tol);
       check = check_flag(&flag, "CVodeSStolerances",RETURNNONNEGATIVE);
       assert(check == 0);
     }

     /// Function to solve the ODE.
     N_Vector
     solve(Real solve_time)
     {
       N_Vector solution = template_vector->ops->nvclone(template_vector);
       Real t;
       flag = CVode(cvode_memory, solve_time, solution, &t, CV_NORMAL);
       const int check = MEPBM::check_flag(&flag, "CVODE",MEPBM::RETURNNONNEGATIVE);
       assert(check == 0);
       return solution;
     }



   private:
     /// Template vector for SUNDIALS to clone when it needs a vector.
     N_Vector template_vector;

     /// Memory structure that SUNDIALS uses for its CVODE solver.
     void * cvode_memory;

     /// Flag for checking if the SUNDIALS functions worked.
     int flag;

     /// The initial time, i.e. the time associated with the provided initial condition (usually time=0).
     const Real initial_time;

     /// The final time, i.e. the last time you want a solution vector.
     const Real final_time;
   };

}


#endif //MEPBM_CVODE_H
