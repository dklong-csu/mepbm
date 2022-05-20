#ifndef MEPBM_AGGLOMERATION_KERNEL_H
#define MEPBM_AGGLOMERATION_KERNEL_H



#include <functional>
#include <vector>



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
}

#endif //MEPBM_AGGLOMERATION_KERNEL_H
