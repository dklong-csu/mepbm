#ifndef MEPBM_IrHPO4_H
#define MEPBM_IrHPO4_H

#include "src/create_nvector.h"


namespace MEPBM {
  namespace IrHPO4 {
    /*
     * To analyse a new nanoparticle system the following is needed
     *
     * 1. Definition of what the mechanisms being tested are.
     * 2. Definition of how particle growth/agglomeration reaction rates are modeled.
     * 3. Definition of how the discrepancy between data and simulation is quantified.
     * 4. Definition of what the starting conditions of the chemical reaction is.
     */

    /********************************************************************************
     * Experimental design
     ********************************************************************************/
     template<typename Vector, typename Real>
     class ExperimentalDesign {
     public:
       ExperimentalDesign()
        : max_size(450),
          solvent_conc(11.7),
          precursor_conc(0.0025),
          hpo4_conc(0.0625)
       {}



       unsigned int max_particle_size() {return max_size;}



       unsigned int vector_length(const unsigned int n_nonparticle_species, const unsigned int first_particle_size) {
         const auto n_particles = max_size - first_particle_size + 1;
         return n_nonparticle_species + n_particles;
       }



       Real IC_solvent() {return solvent_conc;}



       Real IC_precursor() {return precursor_conc;}



       Real IC_hpo4() {return hpo4_conc;}



       unsigned int precursor_index() {return 0;}



       unsigned int hpo4_index() {return 1;}



       Vector* get_vector_pointer(N_Vector vec) {
         auto vec_ptr = static_cast<Vector*>(vec->content);
         return vec_ptr;
       }



      N_Vector IC_vector(const unsigned int n_nonparticle_species, const unsigned int first_particle_size) {
         auto ic = MEPBM::create_eigen_nvector<Vector>(vector_length(n_nonparticle_species, first_particle_size));
         auto ic_ptr = get_vector_pointer(ic);
        (*ic_ptr)(precursor_index()) = IC_precursor();
        (*ic_ptr)(hpo4_index()) = IC_hpo4();
        return ic;
       }
     private:
       const unsigned int max_size;
       const Real solvent_conc;
       const Real precursor_conc;
       const Real hpo4_conc;
     };
    /********************************************************************************
     * Mechanism definitions
     ********************************************************************************/

    /********************************************************************************
     * Growth/Agglomeration functions
     ********************************************************************************/

    /********************************************************************************
     * Discrepancy/Log likelihood models
     ********************************************************************************/
  }
}

#endif //MEPBM_IrHPO4_H
