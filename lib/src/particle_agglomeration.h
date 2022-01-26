#ifndef MEPBM_PARTICLE_AGGLOMERATION_H
#define MEPBM_PARTICLE_AGGLOMERATION_H

#include "chemical_reaction.h"
#include "particle.h"
#include "species.h"


namespace MEPBM {
  /**
     * ParticleAgglomeration represents the process of two particles sticking to each other to form a larger particle.
     * Specifically, the pseudo-elementary step
     *      sum(reactants) + A + B -> C + sum(products)
     * where A and B are particles of potentially different size categories and C is the resultant particle.
     * This leads to the reactions
     *      sum(reactants) + A_i + B_j ->[a_ij * k] C_ij + sum(products)
     */
  template<typename Real, typename Matrix>
  class ParticleAgglomeration
  {
  public:
    ParticleAgglomeration(const Particle particleA,
                          const Particle particleB,
                          const Real reaction_rate,
                          const unsigned int max_particle_size,
                          const std::function<Real(const unsigned int)> growth_kernel,
                          const std::vector<std::pair<Species, unsigned int>> reactants,
                          const std::vector<std::pair<Species, unsigned int>> products)
      : particleA(particleA),
        particleB(particleB),
        reaction_rate(reaction_rate),
        max_particle_size(max_particle_size),
        growth_kernel(growth_kernel),
        reactants(reactants),
        products(products)
    {}

    const Particle particleA, particleB;
    const Real reaction_rate;
    const unsigned int max_particle_size;
    const std::function<Real(const unsigned int)> growth_kernel;
    const std::vector<std::pair<Species, unsigned int>> reactants, products;

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void*)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void * user_data)
      {
        int output_value;

        // Loop through all particleA particles
        for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
        {
          // Extract particleA reactant species
          const auto agglomA = particleA.species(indexA);

          // Loop through all particleB particles
          // FIXME? If particleA=particleB or if there is overlap, then there is some double counting
          // FIXME? But this likely doesn't matter since the inverse problem will just find 1/2*k instead of k
          for (unsigned int indexB=particleB.index_start; indexB <= particleB.index_end; ++indexB)
          {
            // Extract particleB reactant species
            const auto agglomB = particleB.species(indexB);
            auto all_reactants = reactants;
            all_reactants.push_back({agglomA, 1});
            all_reactants.push_back({agglomB, 1});

            auto all_products = products;

            // Check size of created particle and if it's a tracked size, include it as a product.
            const auto sizeA = particleA.size(indexA);
            const auto sizeB = particleB.size(indexB);
            if (sizeA+sizeB <=max_particle_size)
            {
              const auto indexC = particleA.index(sizeA+sizeB);
              const auto agglomC = particleA.species(indexC);
              all_products.push_back({agglomC,1});
            }

            // Calculate the reaction rate.
            const auto rate = reaction_rate * growth_kernel(sizeA) * growth_kernel(sizeB);

            // Create the chemical reaction
            ChemicalReaction<Real, Matrix> rxn(all_reactants, all_products, rate);

            // Get the rhs function
            auto rhs = rxn.rhs_function();

            // Apply the rhs to the inputs
            // FIXME: should this be += instead of =?
            output_value = rhs(time, x, x_dot, user_data);
          }
        }
        return output_value;
      };
      return fcn;
    }

    /// Returns a function for the Jacobian of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, SUNMatrix, void *, N_Vector, N_Vector, N_Vector)>
    jacobian_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, SUNMatrix J, void * user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
      {
        int output_value;

        // Loop through all particleA particles
        for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
        {
          // Extract particleA reactant species
          const auto agglomA = particleA.species(indexA);

          // Loop through all particleB particles
          // FIXME? If particleA=particleB or if there is overlap, then there is some double counting
          // FIXME? But this likely doesn't matter since the inverse problem will just find 1/2*k instead of k
          for (unsigned int indexB=particleB.index_start; indexB <= particleB.index_end; ++indexB)
          {
            // Extract particleB reactant species
            const auto agglomB = particleB.species(indexB);
            auto all_reactants = reactants;
            all_reactants.push_back({agglomA, 1});
            all_reactants.push_back({agglomB, 1});

            auto all_products = products;

            // Check size of created particle and if it's a tracked size, include it as a product.
            const auto sizeA = particleA.size(indexA);
            const auto sizeB = particleB.size(indexB);
            if (sizeA+sizeB <=max_particle_size)
            {
              const auto indexC = particleA.index(sizeA+sizeB);
              const auto agglomC = particleA.species(indexC);
              all_products.push_back({agglomC,1});
            }

            // Calculate the reaction rate.
            const auto rate = reaction_rate * growth_kernel(sizeA) * growth_kernel(sizeB);

            // Create the chemical reaction
            ChemicalReaction<Real, Matrix> rxn(all_reactants, all_products, rate);

            // Get the rhs function
            auto jac = rxn.jacobian_function();

            // Apply the rhs to the inputs
            // FIXME: should this be += instead of =?
            output_value = jac(time, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
          }
        }
        return output_value;
      };
      return fcn;
    }
  };



  /**
     * ParticleAgglomeration represents the process of two particles sticking to each other to form a larger particle.
     * Specifically, the pseudo-elementary step
     *      sum(reactants) + A + B -> C + sum(products)
     * where A and B are particles of potentially different size categories and C is the resultant particle.
     * This leads to the reactions
     *      sum(reactants) + A_i + B_j ->[a_ij * k] C_ij + sum(products)
     * This is a partial specialization for sparse matrices using compressed column storage. The formation of the
     * Jacobian needs to be handled differently than in the dense matrix case.
     */
  template<typename Real>
  class ParticleAgglomeration<Real, Eigen::SparseMatrix<Real>>
  {
  public:
    ParticleAgglomeration(const Particle particleA,
                          const Particle particleB,
                          const Real reaction_rate,
                          const unsigned int max_particle_size,
                          const std::function<Real(const unsigned int)> growth_kernel,
                          const std::vector<std::pair<Species, unsigned int>> reactants,
                          const std::vector<std::pair<Species, unsigned int>> products)
        : particleA(particleA),
          particleB(particleB),
          reaction_rate(reaction_rate),
          max_particle_size(max_particle_size),
          growth_kernel(growth_kernel),
          reactants(reactants),
          products(products)
    {}

    const Particle particleA, particleB;
    const Real reaction_rate;
    const unsigned int max_particle_size;
    const std::function<Real(const unsigned int)> growth_kernel;
    const std::vector<std::pair<Species, unsigned int>> reactants, products;

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void*)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void * user_data)
      {
        int output_value;

        // Loop through all particleA particles
        for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
        {
          // Extract particleA reactant species
          const auto agglomA = particleA.species(indexA);

          // Loop through all particleB particles
          // FIXME? If particleA=particleB or if there is overlap, then there is some double counting
          // FIXME? But this likely doesn't matter since the inverse problem will just find 1/2*k instead of k
          for (unsigned int indexB=particleB.index_start; indexB <= particleB.index_end; ++indexB)
          {
            // Extract particleB reactant species
            const auto agglomB = particleB.species(indexB);
            auto all_reactants = reactants;
            all_reactants.push_back({agglomA, 1});
            all_reactants.push_back({agglomB, 1});

            auto all_products = products;

            // Check size of created particle and if it's a tracked size, include it as a product.
            const auto sizeA = particleA.size(indexA);
            const auto sizeB = particleB.size(indexB);
            if (sizeA+sizeB <=max_particle_size)
            {
              const auto indexC = particleA.index(sizeA+sizeB);
              const auto agglomC = particleA.species(indexC);
              all_products.push_back({agglomC,1});
            }

            // Calculate the reaction rate.
            const auto rate = reaction_rate * growth_kernel(sizeA) * growth_kernel(sizeB);

            // Create the chemical reaction
            ChemicalReaction<Real, Eigen::SparseMatrix<Real>> rxn(all_reactants, all_products, rate);

            // Get the rhs function
            auto rhs = rxn.rhs_function();

            // Apply the rhs to the inputs
            // FIXME: should this be += instead of =?
            output_value = rhs(time, x, x_dot, user_data);
          }
        }
        return output_value;
      };
      return fcn;
    }

    /// Returns a function for the Jacobian of the ODE that is compatible with the SUNDIALS API.
    std::function<int(N_Vector, std::vector<Eigen::Triplet<Real>> &, SUNMatrix)>
    jacobian_function() const {
      auto fcn = [&](N_Vector x, std::vector<Eigen::Triplet<Real>> &triplet_list, SUNMatrix Jacobian)
      {
        int output_value;

        // Loop through all particleA particles
        for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
        {
          // Extract particleA reactant species
          const auto agglomA = particleA.species(indexA);

          // Loop through all particleB particles
          // FIXME? If particleA=particleB or if there is overlap, then there is some double counting
          // FIXME? But this likely doesn't matter since the inverse problem will just find 1/2*k instead of k
          for (unsigned int indexB=particleB.index_start; indexB <= particleB.index_end; ++indexB)
          {
            // Extract particleB reactant species
            const auto agglomB = particleB.species(indexB);
            auto all_reactants = reactants;
            all_reactants.push_back({agglomA, 1});
            all_reactants.push_back({agglomB, 1});

            auto all_products = products;

            // Check size of created particle and if it's a tracked size, include it as a product.
            const auto sizeA = particleA.size(indexA);
            const auto sizeB = particleB.size(indexB);
            if (sizeA+sizeB <=max_particle_size)
            {
              const auto indexC = particleA.index(sizeA+sizeB);
              const auto agglomC = particleA.species(indexC);
              all_products.push_back({agglomC,1});
            }

            // Calculate the reaction rate.
            const auto rate = reaction_rate * growth_kernel(sizeA) * growth_kernel(sizeB);

            // Create the chemical reaction
            ChemicalReaction<Real, Eigen::SparseMatrix<Real>> rxn(all_reactants, all_products, rate);

            // Get the rhs function
            auto jac = rxn.jacobian_function();

            // Apply the rhs to the inputs
            // FIXME: should this be += instead of =?
            output_value = jac(x, triplet_list, Jacobian);
          }
        }
        return output_value;
      };
      return fcn;
    }
  };



  /**
     * ParticleAgglomeration represents the process of two particles sticking to each other to form a larger particle.
     * Specifically, the pseudo-elementary step
     *      sum(reactants) + A + B -> C + sum(products)
     * where A and B are particles of potentially different size categories and C is the resultant particle.
     * This leads to the reactions
     *      sum(reactants) + A_i + B_j ->[a_ij * k] C_ij + sum(products)
     * This is a partial specialization for sparse matrices using compressed row storage. The formation of the
     * Jacobian needs to be handled differently than in the dense matrix case.
     */
  template<typename Real>
  class ParticleAgglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
  {
  public:
    ParticleAgglomeration(const Particle particleA,
                          const Particle particleB,
                          const Real reaction_rate,
                          const unsigned int max_particle_size,
                          const std::function<Real(const unsigned int)> growth_kernel,
                          const std::vector<std::pair<Species, unsigned int>> reactants,
                          const std::vector<std::pair<Species, unsigned int>> products)
        : particleA(particleA),
          particleB(particleB),
          reaction_rate(reaction_rate),
          max_particle_size(max_particle_size),
          growth_kernel(growth_kernel),
          reactants(reactants),
          products(products)
    {}

    const Particle particleA, particleB;
    const Real reaction_rate;
    const unsigned int max_particle_size;
    const std::function<Real(const unsigned int)> growth_kernel;
    const std::vector<std::pair<Species, unsigned int>> reactants, products;

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void*)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void * user_data)
      {
        int output_value;

        // Loop through all particleA particles
        for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
        {
          // Extract particleA reactant species
          const auto agglomA = particleA.species(indexA);

          // Loop through all particleB particles
          // FIXME? If particleA=particleB or if there is overlap, then there is some double counting
          // FIXME? But this likely doesn't matter since the inverse problem will just find 1/2*k instead of k
          for (unsigned int indexB=particleB.index_start; indexB <= particleB.index_end; ++indexB)
          {
            // Extract particleB reactant species
            const auto agglomB = particleB.species(indexB);
            auto all_reactants = reactants;
            all_reactants.push_back({agglomA, 1});
            all_reactants.push_back({agglomB, 1});

            auto all_products = products;

            // Check size of created particle and if it's a tracked size, include it as a product.
            const auto sizeA = particleA.size(indexA);
            const auto sizeB = particleB.size(indexB);
            if (sizeA+sizeB <=max_particle_size)
            {
              const auto indexC = particleA.index(sizeA+sizeB);
              const auto agglomC = particleA.species(indexC);
              all_products.push_back({agglomC,1});
            }

            // Calculate the reaction rate.
            const auto rate = reaction_rate * growth_kernel(sizeA) * growth_kernel(sizeB);

            // Create the chemical reaction
            ChemicalReaction<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> rxn(all_reactants, all_products, rate);

            // Get the rhs function
            auto rhs = rxn.rhs_function();

            // Apply the rhs to the inputs
            // FIXME: should this be += instead of =?
            output_value = rhs(time, x, x_dot, user_data);
          }
        }
        return output_value;
      };
      return fcn;
    }

    /// Returns a function for the Jacobian of the ODE that is compatible with the SUNDIALS API.
    std::function<int(N_Vector, std::vector<Eigen::Triplet<Real>> &, SUNMatrix)>
    jacobian_function() const {
      auto fcn = [&](N_Vector x, std::vector<Eigen::Triplet<Real>> &triplet_list, SUNMatrix Jacobian)
      {
        int output_value;

        // Loop through all particleA particles
        for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
        {
          // Extract particleA reactant species
          const auto agglomA = particleA.species(indexA);

          // Loop through all particleB particles
          // FIXME? If particleA=particleB or if there is overlap, then there is some double counting
          // FIXME? But this likely doesn't matter since the inverse problem will just find 1/2*k instead of k
          for (unsigned int indexB=particleB.index_start; indexB <= particleB.index_end; ++indexB)
          {
            // Extract particleB reactant species
            const auto agglomB = particleB.species(indexB);
            auto all_reactants = reactants;
            all_reactants.push_back({agglomA, 1});
            all_reactants.push_back({agglomB, 1});

            auto all_products = products;

            // Check size of created particle and if it's a tracked size, include it as a product.
            const auto sizeA = particleA.size(indexA);
            const auto sizeB = particleB.size(indexB);
            if (sizeA+sizeB <=max_particle_size)
            {
              const auto indexC = particleA.index(sizeA+sizeB);
              const auto agglomC = particleA.species(indexC);
              all_products.push_back({agglomC,1});
            }

            // Calculate the reaction rate.
            const auto rate = reaction_rate * growth_kernel(sizeA) * growth_kernel(sizeB);

            // Create the chemical reaction
            ChemicalReaction<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> rxn(all_reactants, all_products, rate);

            // Get the rhs function
            auto jac = rxn.jacobian_function();

            // Apply the rhs to the inputs
            // FIXME: should this be += instead of =?
            output_value = jac(x, triplet_list, Jacobian);
          }
        }
        return output_value;
      };
      return fcn;
    }
  };
}
#endif //MEPBM_PARTICLE_AGGLOMERATION_H
