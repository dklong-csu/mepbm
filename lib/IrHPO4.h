#ifndef MEPBM_IrHPO4_H
#define MEPBM_IrHPO4_H

#include "src/create_nvector.h"
#include "src/chemical_reaction_network.h"


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
     // FIXME - add documentation
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



       std::pair<unsigned int, unsigned int> particle_index_range(const unsigned int n_nonparticle_species, const unsigned int first_particle_size) {
         const auto first_index = n_nonparticle_species;
         const auto last_index = vector_length(n_nonparticle_species, first_particle_size) - 1;
         return {first_index, last_index};
       }
     private:
       const unsigned int max_size;
       const Real solvent_conc;
       const Real precursor_conc;
       const Real hpo4_conc;
     };



    /********************************************************************************
    * Growth/Agglomeration functions
    ********************************************************************************/


    /**
     * Each atom in a nanoparticle will be able to bind with other species, so long as it is on the outside
     * of the particle. The reaction thus will be sped up in proportion to how many cluster are on the surface
     * of the particle. Adapted from the work in https://doi.org/10.1007/s11244-005-9261-4.
     *
     * @param size - The particle size
     * @return - The amount the base reaction rate should be multiplied by to account for the particle size.
     */
     template<typename Real>
     Real r_func(const unsigned int size) {
       return (1.0*size) * 2.677 * std::pow(1.0*size, -0.28);
     }



     /**
      * Similar to r_func but caps the regression formula from https://doi.org/10.1007/s11244-005-9261-4 at 1.
      * The original formula from the reference results in values above 1 for small particles sizes. Once
      * multiplied by the size, the interpretation is "how many atoms are on the surface". Giving a number
      * larger than the particle size (i.e. total number of atoms) is unphysical. So capping the number at surface
      * atoms at the total number of atoms has a stronger physical interpretation.
      *
      * @tparam Real - The type representing real-valued numbers (e.g. double, float)
      * @param size  - The particle size
      * @return - The amount the base reaction rate should be multiplied by to account for the particle size.
      */
    template<typename Real>
    Real r_func_capped(const unsigned int size) {
      return std::min(1.0*size, r_func<Real>(size));
    }



    // FIXME - add documentation
    template<typename Real>
    class BackwardsLogisticCurve {
      public:
      BackwardsLogisticCurve(const Real height, const Real midpoint, const Real rate)
        : height(height),
          midpoint(midpoint),
          rate(rate)
        {}


        Real evaluate(const Real x) {
          return height - height / (1 + std::exp(-rate * (x - midpoint)));
        };

      private:
        const Real height;
        const Real midpoint;
        const Real rate;
      };



    // FIXME - add documentation
    template<typename Real, typename Sample>
    class BaseGrowthKernel {
    public:
      // Virtual function gets overriden by derived class. =0 means the derived class MUST define this function.
      virtual std::function<Real(const unsigned int)> get_function(const Sample & sample) = 0;
    };



    // FIXME - add documentation
    template<typename Real, typename Sample>
    class StepGrowthKernel : BaseGrowthKernel<Real, Sample> {
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


      std::function<Real(const unsigned int)> get_function(const Sample & sample) override {
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



    // FIXME - add documentation
    template<typename Real, typename Sample>
    class LogisticCurveGrowthKernel : BaseGrowthKernel<Real, Sample> {
    public:
      LogisticCurveGrowthKernel(std::function<Real(const unsigned int)> calc_surface_atoms,
                                const std::vector<Real> height_values,
                                const std::vector<Real> midpoint_values,
                                const std::vector<Real> rate_values)
        : calc_surface_atoms(calc_surface_atoms),
          height_values(height_values),
          midpoint_values(midpoint_values),
          rate_values(rate_values)
      {
        assert(height_values.size() - 1 == midpoint_values.size());
        assert(midpoint_values.size() == rate_values.size());
      }


    std::function<Real(const unsigned int)> get_function(const Sample & sample) override {
        auto result = [&](const unsigned int size) {
          Real rate = height_values.back();
          for (unsigned int i=0; i<midpoint_values.size(); ++i) {
            const BackwardsLogisticCurve<Real> curve(height_values[i],
                                                     midpoint_values[i],
                                                     rate_values[i]);
            rate += curve.evaluate(size);
          }


          return rate * calc_surface_atoms(size);
        };

        return result;
      }

    private:
      const std::function<Real(const unsigned int)> calc_surface_atoms;
      const std::vector<Real> height_values, midpoint_values, rate_values;

    };




    template<typename Real, typename Sample>
    class BaseAgglomerationKernel {
    public:
      // Virtual function gets overriden by derived class. =0 means the derived class MUST define this function.
      virtual std::function<Real(const unsigned int)> get_function(const Sample & sample) = 0;
    };



    /********************************************************************************
     * Mechanism definitions
     ********************************************************************************/


    /**
     *  Represents the mechanism
     *      A_2 + 2solv     <->[kf,kb]          A_2(solv) + L
     *      A_2(solv)        ->[k1]             B_2 + L
     *      A_2 + B_i        ->[growth(i)]      B_{i+2} + 2L
     *      B_i + B_j        ->[agglom(i,j)]    B_{i+j}
     *  where growth(i) is a function giving the reaction rate for the growth of a particle of size i
     *  and agglom(i,j) is a function giving the reaction rate for agglomeration between a particle of size i and j
     */
    template<typename Vector, typename Matrix, typename Real, typename Sample>
    MEPBM::ChemicalReactionNetwork<Real, Matrix>
    create_mech1A(const Sample & sample,
                  const BaseGrowthKernel<Real, Sample> & growth_kernel,
                  const BaseAgglomerationKernel<Real, Sample> & agglomeration_kernel) {
      // Get the experimental design
      const ExperimentalDesign<Vector, Real> design;


      // Constants specific to the mechanism
      const unsigned int n_nonparticle_species = 3;
      const unsigned int first_particle_size = 2;
      const unsigned int growth_amount = 2;


      // Define chemical species being tracked
      MEPBM::Species A(design.precursor_index());
      MEPBM::Species L(design.hpo4_index());
      MEPBM::Species Asolv(design.hpo4_index()+1);
      const auto particle_index_range = design.particle_index_range(n_nonparticle_species, first_particle_size);
      MEPBM::Particle B(particle_index_range.first, particle_index_range.second, first_particle_size);


      // Extract constants relevant to the reactions
      const auto kf = sample[0];
      const auto kb = sample[1];
      const auto k1 = sample[2];
      const auto S = design.IC_solvent();
      const auto growth_fcn = growth_kernel.get_function(sample);
      const auto agglom_fcn = agglomeration_kernel.get_function(sample);


      // Chemical reactions
      MEPBM::ChemicalReaction<Real, Matrix> nucAf(
          { {A, 1} },
          { {Asolv, 1}, {L, 1} },
          S*S*kf
      );

      MEPBM::ChemicalReaction<Real, Matrix> nucAb(
          { {Asolv, 1}, {L, 1} },
          { {A, 1} },
          kb
      );

      auto B_nuc = B.species(B.index(first_particle_size));
      MEPBM::ChemicalReaction<Real, Matrix> nucB(
          { {Asolv, 1} },
          { {B_nuc, 1}, {L, 1} },
          k1
      );

      MEPBM::ParticleGrowth<Real, Matrix> growth(B,
                                                 growth_amount,
                                                 design.max_particle_size(),
                                                 growth_fcn,
                                                 { {A, 1} },
                                                 { {L, 2}}
      );

      MEPBM::ParticleAgglomeration<Real, Matrix> agglom(B,
                                                        B,
                                                        design.max_particle_size(),
                                                        agglom_fcn,
                                                        {},
                                                        {}
      );

      MEPBM::ChemicalReactionNetwork<Real, Matrix> network({nucAf, nucAb, nucB},
                                                           {growth},
                                                           {agglom});

      return network;
    }



    /********************************************************************************
     * Discrepancy/Log likelihood models
     ********************************************************************************/
  }
}

#endif //MEPBM_IrHPO4_H
