#ifndef MEPBM_AGGLOMERATION_KERNEL_H
#define MEPBM_AGGLOMERATION_KERNEL_H



#include <functional>
#include <vector>
#include <cmath>



namespace MEPBM {
  /// Virtual base class for the agglomeration kernel function
  template<typename Real, typename Sample>
  class BaseAgglomerationKernel {
  public:
    /// Returns a function that can provide the reaction rate of an agglomeration reaction between two particles.
    virtual std::function<Real(const unsigned int, const unsigned int)> get_function(const Sample & sample) const = 0;
  };



  /**
   * An agglomeration kernel based on a step function combined with a modification function (e.g. Schmidt & Smirnov)
   * @tparam Real - The floating point type.
   * @tparam Sample - The type that contains the parameters necessary to create the kernel.
   */
  template<typename Real, typename Sample>
  class StepAgglomerationKernel : public BaseAgglomerationKernel<Real, Sample> {
  public:
    /**
     * Constructor
     * @param calc_surface_atoms - Function used to modify the base growth rate to account for the number of surface atoms.
     * @param sample_indices - the indices in the sample that provide the reaction rate for each step.
     * @param step_locations - the particle sizes corresponding to where a new step takes place.
     */
    StepAgglomerationKernel(std::function<Real(const unsigned int)> calc_surface_atoms,
                            const std::vector<unsigned int> sample_indices,
                            const std::vector<unsigned int> step_locations)
        : calc_surface_atoms(calc_surface_atoms),
          sample_indices(sample_indices),
          step_locations(step_locations)
    {}



    /**
     * Returns a function that is able to calculate the reaction rate for a particle of a specified size.
     * @param sample - the parameters relevant to creating the step function
     * @return
     */
    std::function<Real(const unsigned int, const unsigned int)> get_function(const Sample & sample) const override {
      auto result = [&](const unsigned int sizeA, const unsigned int sizeB) {
        // See what region the two sizes are in
        for (unsigned int i = 0; i<step_locations.size(); ++i) {
          if ( sizeA < step_locations[i] && sizeB < step_locations[i])
            return sample[sample_indices[i]] * calc_surface_atoms(sizeA) * calc_surface_atoms(sizeB);
        }
        // If not, then the last parameter specified in sample provides the base reaction rate
        return sample[sample_indices.back()] * calc_surface_atoms(sizeA) * calc_surface_atoms(sizeB);
      };
      return result;
    }



  private:
    const std::function<Real(const unsigned int)> calc_surface_atoms;
    const std::vector<unsigned int> sample_indices;
    const std::vector<unsigned int> step_locations;
  };



  /**
   * An agglomeration kernel based on a piecewise linear function combined with a modification function (e.g. Schmidt & Smirnov)
   * @tparam Real - The floating point type.
   * @tparam Sample - The type that contains the parameters necessary to create the kernel.
   */
  template<typename Real, typename Sample>
  class PiecewiseLinearAgglomerationKernel : public BaseAgglomerationKernel<Real, Sample> {
  public:
    /**
     * Constructor
     * @param calc_surface_atoms - Function used to modify the base growth rate to account for the number of surface atoms.
     * @param sample_indices - the indices in the sample that provide the reaction rate for each step.
     * @param step_locations - the particle sizes corresponding to where a new step takes place.
     */
    PiecewiseLinearAgglomerationKernel(std::function<Real(const unsigned int)> calc_surface_atoms,
                            const std::vector<unsigned int> sample_indices,
                            const std::vector<unsigned int> step_locations)
        : calc_surface_atoms(calc_surface_atoms),
          sample_indices(sample_indices),
          step_locations(step_locations)
    {}



    /**
     * Returns a function that is able to calculate the reaction rate for a particle of a specified size.
     * @param sample - the parameters relevant to creating the step function
     * @return
     */
    std::function<Real(const unsigned int, const unsigned int)> get_function(const Sample & sample) const override {
      if (step_locations.size() == 1){
        // Global constant
        auto result = [&](const unsigned int sizeA, const unsigned int sizeB){
          return sample[sample_indices[0]] * calc_surface_atoms(sizeA) * calc_surface_atoms(sizeB);
        };
        return result;
      }
      else if (step_locations.size() == 2){
        // Global linear
        auto result = [&](const unsigned int sizeA, const unsigned int sizeB){
          // Function is based on the difference in sizes
          const unsigned int diff = std::abs(1.0*sizeA - 1.0*sizeB);

          const double x1 = step_locations[0];
          const double y1 = sample[sample_indices[0]];
          const double x2 = step_locations[1];
          const double y2 = sample[sample_indices[1]];

          const double m = (y2 - y1)/(x2-x1);

          return (m * (1.0*diff - x1) + y1) * calc_surface_atoms(sizeA) * calc_surface_atoms(sizeB);
        };
        return result;
      }
      else {
        // Piecewise polynomial
        auto result = [&](const unsigned int sizeA, const unsigned int sizeB){
          // Function is based on the difference in sizes
          const unsigned int diff = std::abs(1.0*sizeA - 1.0*sizeB);

          // Bisection to find interval
          unsigned int idx1 = 0;
          unsigned int idx2 = 1;
          if (diff >= step_locations.back()){
            idx1 = step_locations.size()-2;
            idx2 = step_locations.size()-1;
          }
          else if (diff > step_locations[0]){
            unsigned int lo = 0;
            unsigned int hi = step_locations.size()-1;
            while (hi - lo > 1){
              unsigned int mid = (hi + lo)/2;
              if (step_locations[mid] >= diff){
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

          return (m * (1.0*diff - x1) + y1) * calc_surface_atoms(sizeA) * calc_surface_atoms(sizeB);
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

#endif //MEPBM_AGGLOMERATION_KERNEL_H
