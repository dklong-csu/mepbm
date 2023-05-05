#ifndef MEPBM_GROWTH_KERNEL_H
#define MEPBM_GROWTH_KERNEL_H



#include <functional>
#include <vector>
#include <cassert>
#include "src/logistic_curve.h"
#include <iostream>



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



  /**
   * A growth kernel combining a piecewise linear function between different size atoms and a modification function (e.g. Schmidt & Smirnov).
   * Providing one point results in a global constant function (times the modification function).
   * Providing two points results in a global linear function (times the modification function).
   * Providing >=3 points results in a piecewise linear function (times the modification function).
   * @tparam Real - Floating point type
   * @tparam Sample - The type containing a set of parameters
   */
  template<typename Real, typename Sample>
  class PiecewiseLinearGrowthKernel : public BaseGrowthKernel<Real, Sample> {
  public:
    /**
     * Constructor
     * @param calc_surface_atoms - Function used to modify the base growth rate to account for the number of surface atoms.
     * @param sample_indices - the indices in the sample that provide the "y-values" for each point in the piecewise polynomial.
     * @param step_locations - the particle sizes corresponding to the "x-values" for each point in the piecewise polynomial.
     */
    PiecewiseLinearGrowthKernel(std::function<Real(const unsigned int)> calc_surface_atoms,
                     const std::vector<unsigned int> sample_indices,
                     const std::vector<unsigned int> step_locations)
        : calc_surface_atoms(calc_surface_atoms),
          sample_indices(sample_indices),
          step_locations(step_locations)
    {
      assert(sample_indices.size() > 0);
      assert(sample_indices.size() == step_locations.size());
    }



    /**
     * Returns a function that is able to calculate the reaction rate for a particle of a specified size.
     * @param sample - the parameters relevant to creating the step function
     * @return
     */
    std::function<Real(const unsigned int)> get_function(const Sample & sample) const override {
      if (step_locations.size() == 1){
        // Global constant
        auto result = [&](const unsigned int size){
          return sample[sample_indices[0]] * calc_surface_atoms(size);
        };
        return result;
      }
      else if (step_locations.size() == 2){
        // Global linear
        auto result = [&](const unsigned int size){
          const double x1 = step_locations[0];
          const double y1 = sample[sample_indices[0]];
          const double x2 = step_locations[1];
          const double y2 = sample[sample_indices[1]];

          const double m = (y2 - y1)/(x2-x1);

          return (m * (1.0*size - x1) + y1) * calc_surface_atoms(size);
        };
        return result;
      }
      else {
        // Piecewise polynomial
        auto result = [&](const unsigned int size){
          // Bisection to find interval
          unsigned int idx1 = 0;
          unsigned int idx2 = 1;
          if (size >= step_locations.back()){
            idx1 = step_locations.size()-2;
            idx2 = step_locations.size()-1;
          }
          else if (size > step_locations[0]){
            unsigned int lo = 0;
            unsigned int hi = step_locations.size()-1;
            while (hi - lo > 1){
              unsigned int mid = (hi + lo)/2;
              if (step_locations[mid] >= size){
                hi = mid;
              }
              else {
                lo = mid;
              }
            }
            idx1 = lo;
            idx2 = hi;
          }
          // Create linear function
          const double x1 = step_locations[idx1];
          const double y1 = sample[sample_indices[idx1]];
          const double x2 = step_locations[idx2];
          const double y2 = sample[sample_indices[idx2]];

          const double m = (y2 - y1)/(x2-x1);

          return (m * (1.0*size - x1) + y1) * calc_surface_atoms(size);
        };
        return result;
      }
    }

  private:
    const std::function<Real(const unsigned int)> calc_surface_atoms;
    const std::vector<unsigned int> sample_indices;
    const std::vector<unsigned int> step_locations;
  };
}



#endif //MEPBM_GROWTH_KERNEL_H
