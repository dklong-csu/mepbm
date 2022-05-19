#ifndef MEPBM_GROWTH_KERNEL_H
#define MEPBM_GROWTH_KERNEL_H



#include <functional>
#include <vector>
#include <cassert>
#include "src/logistic_curve.h"



namespace MEPBM {
  /// Base class for the growth kernel
  template<typename Real, typename Sample>
  class BaseGrowthKernel {
  public:
    /// Returns a function for the growth kernel.
    virtual std::function<Real(const unsigned int)> get_function(const Sample & sample) const = 0;
  };



  /**
   * A growth kernel combining a step function between different size atoms and a modification function (e.g. Schmidt & Smirnov)
   * @tparam Real - Floating point type
   * @tparam Sample - The type containing a set of parameters
   */
  template<typename Real, typename Sample>
  class StepGrowthKernel : public BaseGrowthKernel<Real, Sample> {
  public:
    /**
     * Constructor
     * @param calc_surface_atoms - Function used to modify the base growth rate to account for the number of surface atoms.
     * @param sample_indices - the indices in the sample that provide the reaction rate for each step.
     * @param step_locations - the particle sizes corresponding to where a new step takes place.
     */
    StepGrowthKernel(std::function<Real(const unsigned int)> calc_surface_atoms,
                     const std::vector<unsigned int> sample_indices,
                     const std::vector<unsigned int> step_locations)
        : calc_surface_atoms(calc_surface_atoms),
          sample_indices(sample_indices),
          step_locations(step_locations)
    {
      assert(sample_indices.size() == step_locations.size() + 1);
    }



    /**
     * Returns a function that is able to calculate the reaction rate for a particle of a specified size.
     * @param sample - the parameters relevant to creating the step function
     * @return
     */
    std::function<Real(const unsigned int)> get_function(const Sample & sample) const override {
      auto result = [&](const unsigned int size) {
        // See if the particle is smaller than any of the specified step locations
        for (unsigned int i = 0; i<step_locations.size(); ++i) {
          if (size < step_locations[i])
            return sample[sample_indices[i]] * calc_surface_atoms(size);
        }
        // If not, then the last parameter specified in sample provides the base reaction rate
        return sample[sample_indices.back()] * calc_surface_atoms(size);
      };
      return result;
    }

  private:
    const std::function<Real(const unsigned int)> calc_surface_atoms;
    const std::vector<unsigned int> sample_indices;
    const std::vector<unsigned int> step_locations;
  };



  /**
   * A growth kernel combining a logistic curve and a modification function (e.g. Schmidt & Smirnov)
   * @tparam Real - Floating point type
   * @tparam Sample - The type containing a set of parameters
   */
  template<typename Real, typename Sample>
  class LogisticCurveGrowthKernel : public BaseGrowthKernel<Real, Sample> {
  public:
    /**
     * Constructor
     * @param calc_surface_atoms - Function used to modify the base growth rate to account for the number of surface atoms.
     * @param height_indices - The indices in the Sample that correspond to the height(s) of the logistic curve(s).
     * @param midpoints  - The particle size values where the logistic curve(s) are centered.
     * @param rate_indices - The indices in the Sample that correspond to the drop rate(s) of the logistic curve(s).
     */
    LogisticCurveGrowthKernel(std::function<Real(const unsigned int)> calc_surface_atoms,
                              const std::vector<unsigned int> height_indices,
                              const std::vector<unsigned int> midpoints,
                              const std::vector<unsigned int> rate_indices)
        : calc_surface_atoms(calc_surface_atoms),
          height_indices(height_indices),
          midpoints(midpoints),
          rate_indices(rate_indices)
    {
      assert(height_indices.size() - 1 == midpoints.size());
      assert(midpoints.size() == rate_indices.size());
    }



    /**
     * Returns a function that can evaluate the rate constant for a particle of a given size.
     * @param sample - The set of parameters that are necessary to create the kernel function.
     * @return
     */
    std::function<Real(const unsigned int)> get_function(const Sample & sample) const override {
      auto result = [&](const unsigned int size) {
        Real rate = sample[height_indices.back()];
        for (unsigned int i=0; i<midpoints.size(); ++i) {
          const BackwardsLogisticCurve<Real> curve(sample[height_indices[i]],
                                                   midpoints[i],
                                                   sample[rate_indices[i]]);
          rate += curve.evaluate(size);
        }


        return rate * calc_surface_atoms(size);
      };

      return result;
    }

  private:
    const std::function<Real(const unsigned int)> calc_surface_atoms;
    const std::vector<unsigned int> height_indices, midpoints, rate_indices;

  };
}

#endif //MEPBM_GROWTH_KERNEL_H
