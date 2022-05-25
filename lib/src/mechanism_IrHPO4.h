#ifndef MEPBM_MECHANISM_IRHPO4_H
#define MEPBM_MECHANISM_IRHPO4_H



#include "src/mechanism.h"
#include "src/experimental_design.h"
#include "src/atoms_to_diameter.h"
#include "src/get_subset.h"


namespace MEPBM {
  namespace IrHPO4 {
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
    class Mech1A : public MEPBM::BaseMechanism<Vector, Matrix, Real, Sample> {
    public:
      Mech1A(const ExperimentalDesign<Vector, Real> & design)
          : design(design)
      {}



      N_Vector make_IC() const override {
        return design.IC_vector(n_nonparticle_species, first_particle_size);
      }



      MEPBM::ChemicalReactionNetwork<Real, Matrix> make_rxns(const Sample & sample,
                                                             const BaseGrowthKernel<Real, Sample> & growth_kernel,
                                                             const BaseAgglomerationKernel<Real, Sample> & agglomeration_kernel) const override {
        // Define chemical species being tracked
        MEPBM::Species A(design.precursor_index());
        MEPBM::Species L(design.ligand_index());
        MEPBM::Species Asolv(design.ligand_index()+1);
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


        // Restricting the particle size for agglomeration MASSIVELY speeds up the ODE solve
        // FIXME: Make the 13 not hard-coded?
        MEPBM::Particle B_agglom(particle_index_range.first, B.index(13), first_particle_size);
        MEPBM::ParticleAgglomeration<Real, Matrix> agglom(B_agglom,
                                                          B_agglom,
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



      Eigen::Matrix<Real, Eigen::Dynamic, 1> extract_particles(const N_Vector x) const override {
        auto particle_index_range = design.particle_index_range(n_nonparticle_species, first_particle_size);
        return MEPBM::get_subset<Real>(x, particle_index_range.first, particle_index_range.second);
      }



      std::vector<Real> get_particle_diameters() const override {
        std::vector<Real> result;
        for (unsigned int i=first_particle_size; i<=design.max_particle_size(); ++i)
          result.push_back(MEPBM::atoms_to_diameter<Real>(i));
        return result;
      }



    private:
      const unsigned int n_nonparticle_species = 3;
      const unsigned int first_particle_size = 2;
      const unsigned int growth_amount = 2;
      const ExperimentalDesign<Vector, Real> design;
    };



    /**
     *  Represents the mechanism
     *      A_2 + 2solv     <->[kf,kb]          A_2(solv) + L
     *      A_2(solv)        ->[k1]             B_2 + L
     *      A_2(solv) + B_i        ->[growth(i)]      B_{i+2} + 2L
     *      B_i + B_j        ->[agglom(i,j)]    B_{i+j}
     *  where growth(i) is a function giving the reaction rate for the growth of a particle of size i
     *  and agglom(i,j) is a function giving the reaction rate for agglomeration between a particle of size i and j
     */
    template<typename Vector, typename Matrix, typename Real, typename Sample>
    class Mech1B : public MEPBM::BaseMechanism<Vector, Matrix, Real, Sample> {
    public:
      Mech1B(const ExperimentalDesign<Vector, Real> & design)
          : design(design)
      {}



      N_Vector make_IC() const override {
        return design.IC_vector(n_nonparticle_species, first_particle_size);
      }



      MEPBM::ChemicalReactionNetwork<Real, Matrix> make_rxns(const Sample & sample,
                                                             const BaseGrowthKernel<Real, Sample> & growth_kernel,
                                                             const BaseAgglomerationKernel<Real, Sample> & agglomeration_kernel) const override {
        // Define chemical species being tracked
        MEPBM::Species A(design.precursor_index());
        MEPBM::Species L(design.ligand_index());
        MEPBM::Species Asolv(design.ligand_index()+1);
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
                                                   { {Asolv, 1} },
                                                   { {L, 2}}
        );


        // Restricting the particle size for agglomeration MASSIVELY speeds up the ODE solve
        // FIXME: Make the 13 not hard-coded?
        MEPBM::Particle B_agglom(particle_index_range.first, B.index(13), first_particle_size);
        MEPBM::ParticleAgglomeration<Real, Matrix> agglom(B_agglom,
                                                          B_agglom,
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



      Eigen::Matrix<Real, Eigen::Dynamic, 1> extract_particles(const N_Vector x) const override {
        auto particle_index_range = design.particle_index_range(n_nonparticle_species, first_particle_size);
        return MEPBM::get_subset<Real>(x, particle_index_range.first, particle_index_range.second);
      }



      std::vector<Real> get_particle_diameters() const override {
        std::vector<Real> result;
        for (unsigned int i=first_particle_size; i<=design.max_particle_size(); ++i)
          result.push_back(MEPBM::atoms_to_diameter<Real>(i));
        return result;
      }



    private:
      const unsigned int n_nonparticle_species = 3;
      const unsigned int first_particle_size = 2;
      const unsigned int growth_amount = 2;
      const ExperimentalDesign<Vector, Real> design;
    };



    /**
     *  Represents the mechanism
     *      A_2 + 4solv     <->[kf,kb]          2A_1(solv) + 2L
     *      2A_1(solv)        ->[k1]             B_2
     *      A_2 + B_i        ->[growth(i)]      B_{i+2} + 2L
     *      B_i + B_j        ->[agglom(i,j)]    B_{i+j}
     *  where growth(i) is a function giving the reaction rate for the growth of a particle of size i
     *  and agglom(i,j) is a function giving the reaction rate for agglomeration between a particle of size i and j
     */
    template<typename Vector, typename Matrix, typename Real, typename Sample>
    class Mech2A : public MEPBM::BaseMechanism<Vector, Matrix, Real, Sample> {
    public:
      Mech2A(const ExperimentalDesign<Vector, Real> & design)
          : design(design)
      {}



      N_Vector make_IC() const override {
        return design.IC_vector(n_nonparticle_species, first_particle_size);
      }



      MEPBM::ChemicalReactionNetwork<Real, Matrix> make_rxns(const Sample & sample,
                                                             const BaseGrowthKernel<Real, Sample> & growth_kernel,
                                                             const BaseAgglomerationKernel<Real, Sample> & agglomeration_kernel) const override {
        // Define chemical species being tracked
        MEPBM::Species A(design.precursor_index());
        MEPBM::Species L(design.ligand_index());
        MEPBM::Species Asolv(design.ligand_index()+1);
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
            { {Asolv, 2}, {L, 2} },
            S*S*S*S*kf
        );

        MEPBM::ChemicalReaction<Real, Matrix> nucAb(
            { {Asolv, 2}, {L, 2} },
            { {A, 1} },
            kb
        );

        auto B_nuc = B.species(B.index(first_particle_size));
        MEPBM::ChemicalReaction<Real, Matrix> nucB(
            { {Asolv, 2} },
            { {B_nuc, 1} },
            k1
        );

        MEPBM::ParticleGrowth<Real, Matrix> growth(B,
                                                   growth_amount,
                                                   design.max_particle_size(),
                                                   growth_fcn,
                                                   { {A, 1} },
                                                   { {L, 2}}
        );


        // Restricting the particle size for agglomeration MASSIVELY speeds up the ODE solve
        // FIXME: Make the 13 not hard-coded?
        MEPBM::Particle B_agglom(particle_index_range.first, B.index(13), first_particle_size);
        MEPBM::ParticleAgglomeration<Real, Matrix> agglom(B_agglom,
                                                          B_agglom,
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



      Eigen::Matrix<Real, Eigen::Dynamic, 1> extract_particles(const N_Vector x) const override {
        auto particle_index_range = design.particle_index_range(n_nonparticle_species, first_particle_size);
        return MEPBM::get_subset<Real>(x, particle_index_range.first, particle_index_range.second);
      }



      std::vector<Real> get_particle_diameters() const override {
        std::vector<Real> result;
        for (unsigned int i=first_particle_size; i<=design.max_particle_size(); ++i)
          result.push_back(MEPBM::atoms_to_diameter<Real>(i));
        return result;
      }



    private:
      const unsigned int n_nonparticle_species = 3;
      const unsigned int first_particle_size = 2;
      const unsigned int growth_amount = 2;
      const ExperimentalDesign<Vector, Real> design;
    };



    /**
     *  Represents the mechanism
     *      A_2 + 4solv     <->[kf,kb]          2A_1(solv) + 2L
     *      2A_1(solv)        ->[k1]             B_2
     *      A_1(solv) + B_i        ->[growth(i)]      B_{i+1}
     *      B_i + B_j        ->[agglom(i,j)]    B_{i+j}
     *  where growth(i) is a function giving the reaction rate for the growth of a particle of size i
     *  and agglom(i,j) is a function giving the reaction rate for agglomeration between a particle of size i and j
     */
    template<typename Vector, typename Matrix, typename Real, typename Sample>
    class Mech2B : public MEPBM::BaseMechanism<Vector, Matrix, Real, Sample> {
    public:
      Mech2B(const ExperimentalDesign<Vector, Real> & design)
          : design(design)
      {}



      N_Vector make_IC() const override {
        return design.IC_vector(n_nonparticle_species, first_particle_size);
      }



      MEPBM::ChemicalReactionNetwork<Real, Matrix> make_rxns(const Sample & sample,
                                                             const BaseGrowthKernel<Real, Sample> & growth_kernel,
                                                             const BaseAgglomerationKernel<Real, Sample> & agglomeration_kernel) const override {
        // Define chemical species being tracked
        MEPBM::Species A(design.precursor_index());
        MEPBM::Species L(design.ligand_index());
        MEPBM::Species Asolv(design.ligand_index()+1);
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
            { {Asolv, 2}, {L, 2} },
            S*S*S*S*kf
        );

        MEPBM::ChemicalReaction<Real, Matrix> nucAb(
            { {Asolv, 2}, {L, 2} },
            { {A, 1} },
            kb
        );

        auto B_nuc = B.species(B.index(first_particle_size));
        MEPBM::ChemicalReaction<Real, Matrix> nucB(
            { {Asolv, 2} },
            { {B_nuc, 1} },
            k1
        );

        MEPBM::ParticleGrowth<Real, Matrix> growth(B,
                                                   growth_amount,
                                                   design.max_particle_size(),
                                                   growth_fcn,
                                                   { {Asolv, 1} },
                                                   { }
        );


        // Restricting the particle size for agglomeration MASSIVELY speeds up the ODE solve
        // FIXME: Make the 13 not hard-coded?
        MEPBM::Particle B_agglom(particle_index_range.first, B.index(13), first_particle_size);
        MEPBM::ParticleAgglomeration<Real, Matrix> agglom(B_agglom,
                                                          B_agglom,
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



      Eigen::Matrix<Real, Eigen::Dynamic, 1> extract_particles(const N_Vector x) const override {
        auto particle_index_range = design.particle_index_range(n_nonparticle_species, first_particle_size);
        return MEPBM::get_subset<Real>(x, particle_index_range.first, particle_index_range.second);
      }



      std::vector<Real> get_particle_diameters() const override {
        std::vector<Real> result;
        for (unsigned int i=first_particle_size; i<=design.max_particle_size(); ++i)
          result.push_back(MEPBM::atoms_to_diameter<Real>(i));
        return result;
      }



    private:
      const unsigned int n_nonparticle_species = 3;
      const unsigned int first_particle_size = 2;
      const unsigned int growth_amount = 1;
      const ExperimentalDesign<Vector, Real> design;
    };
  }
}
#endif //MEPBM_MECHANISM_IRHPO4_H
