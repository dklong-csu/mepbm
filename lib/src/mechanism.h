#ifndef MEPBM_MECHANISM_H
#define MEPBM_MECHANISM_H



#include "src/growth_kernel.h"
#include "src/agglomeration_kernel.h"
#include "src/chemical_reaction_network.h"



namespace MEPBM {
  /**
   * Virtual base class for a mechanism.
   * @tparam Vector - A class representing a linear algebra vector.
   * @tparam Matrix - A class representing a linear algebra matrix.
   * @tparam Real - Floating point type
   * @tparam Sample - Object containing the parameters needed to construct the chemical reactions.
   */
  template<typename Vector, typename Matrix, typename Real, typename Sample>
  class BaseMechanism {
  public:
    /// Makes the initial condition vector
    virtual N_Vector make_IC() const = 0;

    /// Makes the network of chemical reactions
    virtual MEPBM::ChemicalReactionNetwork<Real, Matrix> make_rxns(const Sample & sample,
                                                                   const BaseGrowthKernel<Real, Sample> & growth_kernel,
                                                                   const BaseAgglomerationKernel<Real, Sample> & agglomeration_kernel) const = 0;

    /// Isolates the particles associated with a mechanism from an `N_Vector`
    virtual Eigen::Matrix<Real, Eigen::Dynamic, 1> extract_particles(const N_Vector x) const = 0;

    /// Converts each particle size to a diameter
    virtual std::vector<Real> get_particle_diameters() const = 0;
  };
}

#endif //MEPBM_MECHANISM_H
