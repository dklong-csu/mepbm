#ifndef MEPBM_EXPERIMENTAL_DESIGN_H
#define MEPBM_EXPERIMENTAL_DESIGN_H


#include "src/create_nvector.h"
#include <iostream>


namespace MEPBM {
  /**
   * A class that provides a description of the experimental setup such as physical constants and relevant vectors.
   * @tparam Vector
   * @tparam Real
   */
  template<typename Vector, typename Real>
  class ExperimentalDesign {
  public:
    /**
     * Default constructor. Using values of 0 for the member variables should break
     */
    ExperimentalDesign()
        : max_size(0),
          solvent_conc(0),
          precursor_conc(0),
          ligand_conc(0),
          final_time(0)
    {
      std::cout << std::endl << "WARNING: For real applications, ExperimentalDesign needs to use a non-default constructor." << std::endl;
    }



    /**
     * Constructor assigning values for all experimental constants.
     * @param max_size - Maximum particle size allowed.
     * @param solvent_conc - Concentration of the Solvent used in the reaction (assumed constant throughout the reaction).
     * @param precursor_conc - Concentration of the precursor species at the start of the reaction.
     * @param ligand_conc - Concentration of the ligand species at the start of the reaction.
     * @param final_time - The end time of the experiment.
     */
    ExperimentalDesign(const unsigned int max_size,
                       const Real solvent_conc,
                       const Real precursor_conc,
                       const Real ligand_conc,
                       const Real final_time)
      : max_size(max_size),
        solvent_conc(solvent_conc),
        precursor_conc(precursor_conc),
        ligand_conc(ligand_conc),
        final_time(final_time)
    {}



    /// Returns the maximum particle size of the system
    unsigned int max_particle_size() const {return max_size;}



    /**
     * Returns the length of the vector associated with this chemical system.
     * @param n_nonparticle_species - The number of species considered that are not particles.
     * @param first_particle_size - The size of the first particle created in nucleation.
     * @return
     */
    unsigned int vector_length(const unsigned int n_nonparticle_species, const unsigned int first_particle_size) const {
      const auto n_particles = max_size - first_particle_size + 1;
      return n_nonparticle_species + n_particles;
    }



    /// Returns the concentration of the solvent in the experiment.
    Real IC_solvent() const {return solvent_conc;}



    /// Returns the initial concentration of the precursor species.
    Real IC_precursor() const {return precursor_conc;}



    /// Returns the initial concentration of the ligand species.
    Real IC_ligand() const {return ligand_conc;}



    /// Returns the vector index for the precursor species.
    unsigned int precursor_index() const {return 0;}



    /// Returns the vector index for the ligand species.
    unsigned int ligand_index() const {return 1;}



    /// Function that returns the underlying vector pointer so that the vector can be easily modified.
    Vector* get_vector_pointer(N_Vector vec) const {
      auto vec_ptr = static_cast<Vector*>(vec->content);
      return vec_ptr;
    }



    /**
     * Returns the vector that describes the initial concentrations of each chemical species and particle.
     * @param n_nonparticle_species - The number of species considered that are not particles.
     * @param first_particle_size - The size of the first particle created in nucleation.
     * @return
     */
    N_Vector IC_vector(const unsigned int n_nonparticle_species, const unsigned int first_particle_size) const {
      auto ic = MEPBM::create_eigen_nvector<Vector>(vector_length(n_nonparticle_species, first_particle_size));
      auto ic_ptr = get_vector_pointer(ic);
      (*ic_ptr)(precursor_index()) = IC_precursor();
      (*ic_ptr)(ligand_index()) = IC_ligand();
      return ic;
    }



    /**
     * Returns the first and last vector indices that correspond to particles.
     * @param n_nonparticle_species - The number of species considered that are not particles.
     * @param first_particle_size - The size of the first particle created in nucleation.
     * @return
     */
    std::pair<unsigned int, unsigned int> particle_index_range(const unsigned int n_nonparticle_species, const unsigned int first_particle_size) const {
      const auto first_index = n_nonparticle_species;
      const auto last_index = vector_length(n_nonparticle_species, first_particle_size) - 1;
      return {first_index, last_index};
    }



    /// Returns the end time of the reaction.
    Real get_end_time() {return final_time;}



  private:
    const unsigned int max_size;
    const Real solvent_conc;
    const Real precursor_conc;
    const Real ligand_conc;
    const Real final_time;

  };
}
#endif //MEPBM_EXPERIMENTAL_DESIGN_H
