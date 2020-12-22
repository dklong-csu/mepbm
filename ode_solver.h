#ifndef MEPBM_ODE_SOLVER_H
#define MEPBM_ODE_SOLVER_H


namespace ODE
{
  /*
   * A function which solves an ODE in an abstract way. The details of how to proceed from one time step to the next
   * is left up to the input.
   */
  template<class OdeSystemType, class VectorType>
  VectorType solve_ode(const OdeSystemType &ode_system, const VectorType &ic,
                       const double start_time, const double end_time, double dt)
  {
    // Check for the pathological case where only 1 time step is used and make sure the time step is appropriate.
    if (start_time + dt > end_time)
      dt = end_time - start_time;

    // x0 is going to represent the current solution
    // x1 is going to represent the next solution
    // Here, these two are instantiated for use later and the first ``current solution" needs to be
    // the initial condition
    auto x0 = ic;
    VectorType x1;

    // t is used to track the current time and it needs to start at the specified start time.
    double t = start_time;
    while (t < end_time)
    {
      // OdeSystemType needs to have a step_forward method which describes how the next time step is
      // solved for -- e.g. implicit Euler method.
      x1 = ode_system.step_forward(x0, t, dt);

      // If the next time step would go past the ending time, adjust the time step to end exactly on the end time.
      if (t + dt > end_time)
        dt = end_time - t;

      // Update the time and the current solution.
      t += dt;
      x0 = x1;
    }

    return x1
  }
}

#endif //MEPBM_ODE_SOLVER_H