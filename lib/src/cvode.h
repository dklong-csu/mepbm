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
#include <cstdio>
#include <utility>



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
           Real final_time,
           const int max_steps=-1)
     : template_vector(nullptr),
       cvode_memory(nullptr),
       flag(-1)
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

       // Set error file
       err_file = fopen("SUNDIALS_error_log.txt","a");
       flag = CVodeSetErrFile(cvode_memory, err_file);
       check = check_flag(&flag, "CVodeSetErrFile", MEMORY);
       assert(check == 0);


       // Specify tolerances -- these can be modified with the `set_tolerance()` function prior to calling `solve()`
       // Based on my experience with the equations that arise in MEPBM applications, these default tolerances are pretty good.
       flag = CVodeSStolerances(cvode_memory, 1e-6, 1e-12);
       check = check_flag(&flag, "CVodeSStolerances",RETURNNONNEGATIVE);
       assert(check == 0);


       // Set maximum number of steps
       flag = CVodeSetMaxNumSteps(cvode_memory, max_steps);
       check = check_flag(&flag, "CVodeSetMaxNumSteps", RETURNNONNEGATIVE);
       assert(check == 0);


       // Activate the BDF stability limit detection algorithm
       flag = CVodeSetStabLimDet(cvode_memory, SUNTRUE);
       check = check_flag(&flag, "CVodeSetStabLimDet", RETURNZERO);
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
       fclose(err_file);
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

     /// Function to set user_data that is needed for the right-hand side or Jacobian functions
     void
     set_user_data(void * user_data)
     {
       int check = 0;
       flag = CVodeSetUserData(cvode_memory, user_data);
       check = check_flag(&flag, "CVodeSetUserData", RETURNZERO);
       assert(check==0);
     }

     /// Function to solve the ODE.
     std::pair<N_Vector, int>
     solve(Real solve_time)
     {
       N_Vector solution = template_vector->ops->nvclone(template_vector);
       Real t;
       flag = CVode(cvode_memory, solve_time, solution, &t, CV_NORMAL);
       const int check = MEPBM::check_flag(&flag, "CVODE",MEPBM::RETURNNONNEGATIVE);
       //std::cout << flag << std::endl;
       assert(check == 0);
       return {solution, check};
     }



   private:
     /// Template vector for SUNDIALS to clone when it needs a vector.
     N_Vector template_vector;

     /// Memory structure that SUNDIALS uses for its CVODE solver.
     void * cvode_memory;

     /// Flag for checking if the SUNDIALS functions worked.
     int flag;

     /// Error file
     FILE* err_file;
   };

}


#endif //MEPBM_CVODE_H
