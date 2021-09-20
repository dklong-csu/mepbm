#ifndef MEPBM_SAMPLE_H
#define MEPBM_SAMPLE_H

#include "nvector_eigen.h"
#include "sunmatrix_eigen.h"
#include "models.h"
#include <vector>
#include "histogram.h"
#include <functional>
#include <utility>
#include <stdexcept>

namespace Sampling
{
  /**
   * An object holding all of the member variables a Sample needs that stay constant during the sampling procedure.
   * In other words, this object holds all necessary member variables except for the model parameters.
   */
  template<typename RealType, typename Matrix>
  class ModelingParameters
  {
  public:
    ModelingParameters(std::vector< std::pair<RealType, RealType> > real_prm_bounds,
                       std::vector< std::pair<int, int> > int_prm_bounds,
                       std::function< Model::Model<RealType, Matrix>(const std::vector<RealType>, const std::vector<int>) > create_model_fcn,
                       N_Vector initial_condition,
                       RealType start_time,
                       RealType end_time,
                       RealType abs_tol,
                       RealType rel_tol,
                       std::vector<RealType> times,
                       unsigned int first_particle_index,
                       unsigned int last_particle_index,
                       Histograms::Parameters<RealType> binning_parameters,
                       unsigned int first_particle_size,
                       unsigned int particle_size_increase,
                       std::vector< std::vector<RealType> > data);



    /// Bounds on each real-valued parameter, together with integer_parameter_bounds creates the domain.
    const std::vector< std::pair<RealType, RealType> > real_parameter_bounds;

    /// Bounds on each integer-valued parameter, together with real_parameter_bounds creates the domain.
    const std::vector< std::pair<int, int> > integer_parameter_bounds;

    /**
     * A user-supplied function that defines what ODE model is being used. Used to create ode_system in the constructor.
     * Intended to take in a std::vector for real_valued_parameters and integer_valued_parameters where the order in the
     * vector is consistent across all member variables with a Sample.
     */
    const std::function< Model::Model<RealType, Matrix>(const std::vector<RealType>, const std::vector<int>) > create_model;

    /// A vector containing the initial condition of the ODE model associated with this Sample.
    const N_Vector initial_condition;

    /// The start time for the ODE.
    const RealType start_time;

    /// The end time for the ODE.
    const RealType end_time;

    /// The absolute tolerance to be used in the ODE solver.
    const RealType abs_tol;

    /// The relative tolerance to be used in the ODE solver.
    const RealType rel_tol;

    /// The times for which data is recorded and therefore the times for which the solution to the ODE is calculated.
    const std::vector<RealType> times;

    /**
     * The index in the N_Vector's corresponding to the first species that is a particle.
     * It is assumed that the N_Vector is organized in such a way that all species considered particles are stored
     * contiguously.
     */
    const unsigned int first_particle_index;

    /**
     * The index in the N_Vector's corresponding to the last species that is a particle.
     * It is assumed that the N_Vector is organized in such a way that all species considered particles are stored
     * contiguously.
     */
    const unsigned int last_particle_index;

    /// The parameters that determine how the data and the ODE solution are binned for computing the likelihood.
    const Histograms::Parameters<RealType> binning_parameters;

    /// The number of atoms present in the first particle.
    const unsigned int first_particle_size;

    /// The amount of additional atoms in particle \f$i+1\f$ versus particle \f$i\f$.
    const unsigned int particle_size_increase;

    /// A vector containing the data to be used in the likelihood calculation.
    std::vector< std::vector<RealType> > data;
  };



  template<typename RealType, typename Matrix>
  ModelingParameters<RealType, Matrix>::ModelingParameters(
    const std::vector< std::pair<RealType, RealType> > real_prm_bounds,
    const std::vector< std::pair<int, int> > int_prm_bounds,
    const std::function< Model::Model<RealType, Matrix>(const std::vector<RealType>, const std::vector<int>) > create_model_fcn,
    const N_Vector initial_condition,
    const RealType start_time,
    const RealType end_time,
    const RealType abs_tol,
    const RealType rel_tol,
    const std::vector<RealType> times,
    const unsigned int first_particle_index,
    const unsigned int last_particle_index,
    const Histograms::Parameters<RealType> binning_parameters,
    const unsigned int first_particle_size,
    const unsigned int particle_size_increase,
    const std::vector< std::vector<RealType> > data)
      : real_parameter_bounds(real_prm_bounds),
        integer_parameter_bounds(int_prm_bounds),
        create_model(create_model_fcn),
        initial_condition(initial_condition),
        start_time(start_time),
        end_time(end_time),
        abs_tol(abs_tol),
        rel_tol(rel_tol),
        times(times),
        first_particle_index(first_particle_index),
        last_particle_index(last_particle_index),
        binning_parameters(binning_parameters),
        first_particle_size(first_particle_size),
        particle_size_increase(particle_size_increase),
        data(data)
  {}



  /**
   * A Sample is a data structure that describes the adjustable parameters in a model as well as the necessary
   * information to compute a likelihood of the sample.
   */
  template<typename RealType, typename Matrix>
  class Sample
  {
  public:
    /**
     * A constructor taking in the necessary arguments to fully define a sample in the desired parameter space.
     */
    Sample(std::vector<RealType> real_prm,
           std::vector<int> int_prm,
           ModelingParameters<RealType, Matrix> user_data);



    /**
     * A function that checks if a set of parameters is within the domain of the parameter space.
     */
     bool
     check_all_parameters_within_bounds();



     /**
      * A function that creates a copy of this Sample except changes the parameters based on the user input.
      */
     Sample<RealType, Matrix>
     create_new_sample(const std::vector<RealType> real_parameters,
                       const std::vector<int> integer_parameters);

    /// The values of all the real-valued parameters.
    const std::vector<RealType> real_valued_parameters;

    /// The values of all the integer-valued parameters.
    const std::vector<int> integer_valued_parameters;

    /// All other necessary parameters that stay constant during the sampling procedure.
    const ModelingParameters<RealType, Matrix> user_data;

    /// The system of ODEs describing the chemical reactions.
    Model::Model<RealType, Matrix> ode_system;
  };



  template<typename RealType, typename Matrix>
  Sample<RealType, Matrix>::Sample(const std::vector<RealType> real_prm,
                                   const std::vector<int> int_prm,
                                   const ModelingParameters<RealType, Matrix> settings)
   : real_valued_parameters(real_prm),
     integer_valued_parameters(int_prm),
     user_data(settings)
  {
    // Use the provided create_model function to create the ODE system
    ode_system = user_data.create_model(real_valued_parameters,
                                        integer_valued_parameters);
  }



  template<typename RealType, typename Matrix>
  bool
  Sample<RealType, Matrix>::check_all_parameters_within_bounds()
  {
    if (real_valued_parameters.size() != user_data.real_parameter_bounds.size())
      throw std::domain_error("The number of real-valued parameters being checked is not equal to the number of parameter bounds.");

    if (integer_valued_parameters.size() != user_data.integer_parameter_bounds.size())
      throw std::domain_error("The number of integer-valued parameters being checked is not equal to the number of parameter bounds.");

    for (unsigned int i=0; i<real_valued_parameters.size(); ++i)
    {
      if (real_valued_parameters[i] < user_data.real_parameter_bounds[i].first
          || real_valued_parameters[i] > user_data.real_parameter_bounds[i].second)
        return false;
    }

    for (unsigned int i=0; i<integer_valued_parameters.size(); ++i)
    {
      if (integer_valued_parameters[i] < user_data.integer_parameter_bounds[i].first
          || integer_valued_parameters[i] > user_data.integer_parameter_bounds[i].second)
        return false;
    }

    return true;
  }



  /**
   * A function that creates a copy of this Sample except changes the parameters based on the user input.
   */
  template<typename RealType, typename Matrix>
  Sample<RealType, Matrix>
  Sample<RealType, Matrix>::create_new_sample(const std::vector<RealType> real_parameters,
                                              const std::vector<int> integer_parameters)
  {
    Sample<RealType, Matrix> new_sample(real_parameters,
                                        integer_parameters,
                                        user_data);
    return new_sample;
  }
}

#endif //MEPBM_SAMPLE_H
