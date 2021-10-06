#ifndef MEPBM_SAMPLING_PARAMETERS_H
#define MEPBM_SAMPLING_PARAMETERS_H

#include "nvector_eigen.h"
#include <functional>
#include "histogram.h"
#include <vector>
#include "models.h"
#include <utility>

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
    /// Default constructor
    ModelingParameters();

    /// Constructor
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

    /// Copy constructor
    ModelingParameters(const ModelingParameters & old_prm);

    /// Copy assignment
    ModelingParameters & operator=(const ModelingParameters& old_prm);

    /// Destructor
    ~ModelingParameters();


    /// Bounds on each real-valued parameter, together with integer_parameter_bounds creates the domain.
    std::vector< std::pair<RealType, RealType> > real_parameter_bounds;

    /// Bounds on each integer-valued parameter, together with real_parameter_bounds creates the domain.
    std::vector< std::pair<int, int> > integer_parameter_bounds;

    /**
     * A user-supplied function that defines what ODE model is being used. Used to create ode_system in the constructor.
     * Intended to take in a std::vector for real_valued_parameters and integer_valued_parameters where the order in the
     * vector is consistent across all member variables with a Sample.
     */
    std::function< Model::Model<RealType, Matrix>(const std::vector<RealType>, const std::vector<int>) > create_model;

    /// A vector containing the initial condition of the ODE model associated with this Sample.
    N_Vector initial_condition;

    /// The start time for the ODE.
    RealType start_time;

    /// The end time for the ODE.
    RealType end_time;

    /// The absolute tolerance to be used in the ODE solver.
    RealType abs_tol;

    /// The relative tolerance to be used in the ODE solver.
    RealType rel_tol;

    /// The times for which data is recorded and therefore the times for which the solution to the ODE is calculated.
    std::vector<RealType> times;

    /**
     * The index in the N_Vector's corresponding to the first species that is a particle.
     * It is assumed that the N_Vector is organized in such a way that all species considered particles are stored
     * contiguously.
     */
    unsigned int first_particle_index;

    /**
     * The index in the N_Vector's corresponding to the last species that is a particle.
     * It is assumed that the N_Vector is organized in such a way that all species considered particles are stored
     * contiguously.
     */
    unsigned int last_particle_index;

    /// The parameters that determine how the data and the ODE solution are binned for computing the likelihood.
    Histograms::Parameters<RealType> binning_parameters;

    /// The number of atoms present in the first particle.
    unsigned int first_particle_size;

    /// The amount of additional atoms in particle \f$i+1\f$ versus particle \f$i\f$.
    unsigned int particle_size_increase;

    /// A vector containing the data to be used in the likelihood calculation.
    std::vector< std::vector<RealType> > data;
  };



  template<typename RealType, typename Matrix>
  ModelingParameters<RealType, Matrix>::ModelingParameters()
  {
    initial_condition = nullptr;
  }



  template<typename RealType, typename Matrix>
  ModelingParameters<RealType, Matrix>::ModelingParameters(
      const std::vector< std::pair<RealType, RealType> > real_prm_bounds,
      const std::vector< std::pair<int, int> > int_prm_bounds,
      const std::function< Model::Model<RealType, Matrix>(const std::vector<RealType>, const std::vector<int>) > create_model_fcn,
      const N_Vector ic,
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
        initial_condition(nullptr),
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
  {
    initial_condition = ic->ops->nvclone(ic);
    auto new_ic_ptr = static_cast<Eigen::Matrix<RealType, Eigen::Dynamic, 1>*>(initial_condition->content);
    auto old_ic_ptr = static_cast<Eigen::Matrix<RealType, Eigen::Dynamic, 1>*>(ic->content);
    assert(new_ic_ptr->size() == old_ic_ptr->size());
    assert(new_ic_ptr->size() == initial_condition->ops->nvgetlength(initial_condition));
    for (unsigned int i=0; i<initial_condition->ops->nvgetlength(initial_condition);++i)
    {
      (*new_ic_ptr)(i) = (*old_ic_ptr)(i);
    }
  }



  template<typename RealType, typename Matrix>
  ModelingParameters<RealType,Matrix>::ModelingParameters(const ModelingParameters<RealType,Matrix> & old_prm)
    : ModelingParameters(old_prm.real_parameter_bounds,
                         old_prm.integer_parameter_bounds,
                         old_prm.create_model,
                         old_prm.initial_condition,
                         old_prm.start_time,
                         old_prm.end_time,
                         old_prm.abs_tol,
                         old_prm.rel_tol,
                         old_prm.times,
                         old_prm.first_particle_index,
                         old_prm.last_particle_index,
                         old_prm.binning_parameters,
                         old_prm.first_particle_size,
                         old_prm.particle_size_increase,
                         old_prm.data)
  {}



  template<typename RealType, typename Matrix>
  ModelingParameters<RealType, Matrix> &
  ModelingParameters<RealType, Matrix>::operator=(const ModelingParameters<RealType, Matrix>& old_prm)
  {
    // Member variables that are not points can be directly copy assigned
    real_parameter_bounds = old_prm.real_parameter_bounds;
    integer_parameter_bounds = old_prm.integer_parameter_bounds;
    create_model = old_prm.create_model;
    start_time = old_prm.start_time;
    end_time = old_prm.end_time;
    abs_tol = old_prm.abs_tol;
    rel_tol = old_prm.rel_tol;
    times = old_prm.times;
    first_particle_index = old_prm.first_particle_index;
    last_particle_index = old_prm.last_particle_index;
    binning_parameters = old_prm.binning_parameters;
    first_particle_size = old_prm.first_particle_size;
    particle_size_increase = old_prm.particle_size_increase;
    data = old_prm.data;

    // N_Vector member needs to be created specially
    initial_condition->ops->nvdestroy(initial_condition);
    initial_condition = old_prm.initial_condition->ops->nvclone(old_prm.initial_condition);
    auto new_ic_ptr = static_cast<Eigen::Matrix<RealType, Eigen::Dynamic, 1>*>(initial_condition->content);
    auto old_ic_ptr = static_cast<Eigen::Matrix<RealType, Eigen::Dynamic, 1>*>(old_prm.initial_condition->content);
    // assert both Eigen vectors have same length
    abort();
    assert(new_ic_ptr->size() == old_ic_ptr->size());
    assert(new_ic_ptr->size() == initial_condition->ops->nvgetlength(initial_condition)+1);
    for (unsigned int i=0; i<initial_condition->ops->nvgetlength(initial_condition); ++i)
    {
      (*new_ic_ptr)(i) = (*old_ic_ptr)(i);
    }

    return *this;
  }



  template<typename RealType, typename Matrix>
  ModelingParameters<RealType, Matrix>::~ModelingParameters()
  {
    initial_condition->ops->nvdestroy(initial_condition);
  }
}






#endif //MEPBM_SAMPLING_PARAMETERS_H
