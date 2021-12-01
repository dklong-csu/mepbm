#ifndef MEPBM_CHEMICAL_REACTION_H
#define MEPBM_CHEMICAL_REACTION_H


#include <vector>
#include <utility>
#include <cmath>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "models.h"
#include <functional>
#include <iostream>



namespace Model
{
  /**
   * A Species represents a chemical species that is being tracked. It is simply a way to give indices an explicit meaning.
   */
   class Species
   {
   public:
     Species(const unsigned int index);

     const unsigned int index;
   };



   Species::Species(const unsigned int index)
    : index(index)
   {}



   /**
    * A Particle represents the set of indices that span a set of particles of contiguous size. The size of the particle
    * is indicated as well as a way to work with a particle of a particular size as a Species.
    */
    class Particle
    {
    public:
      /// Defines a particle by listing the start and end index the particle represents in a vector and the size of the first particle.
      Particle(const unsigned int index_start, const unsigned int index_end, const unsigned int first_size);

      const unsigned int index_start;
      const unsigned int index_end;
      const unsigned int first_size;

      /// Given a vector index, a Species representing the particle with that index is returned.
      Species get_particle_species(const unsigned int index) const;
    };



    Particle::Particle(const unsigned int index_start, const unsigned int index_end, const unsigned int first_size)
      : index_start(index_start), index_end(index_end), first_size(first_size)
    {}



    Species Particle::get_particle_species(const unsigned int index) const
    {
        return Species(index);
    }



    /**
     * A ChemicalReaction is the representation of a chemical reaction's contribution to the system of ODEs.
     * For a list of reactants, products, and the reaction rate, this class represents the reaction
     *      reactants = { [R_1, cr_1], [R_2, cr_2], ... , [R_n, cr_n] }
     *      products  = { [P_1, cp_1], [P_2, cp_2], ... , [P_m, cp_m] }
     *      reaction rate = k
     *
     *      cr_1*R_1 + cr_2*R_2 + ... + cr_n*R_n ->[k] cp_1*P_1 + cp_2*P_2 + ... + cp_m*P_m
     */
     template <typename Real, typename Matrix>
     class ChemicalReaction : public RightHandSideContribution<Real, Matrix>
     {
     public:
       /**
        * Constructor taking in a list of all chemical species, their coefficients, and the reaction rate.
        * @param[in] reactants Each chemical species that reacts as well as their multiplicity.
        * @param[in] products Each chemical species that is produced as well as their multiplicity.
        * @param[in] reaction_rate The rate at which the reaction takes place. Include constant concentration species in this quantity instead of in the reactants.
        */
       ChemicalReaction(const std::vector<std::pair<Species, unsigned int>> reactants,
                        const std::vector<std::pair<Species, unsigned int>> products,
                        const Real reaction_rate);

       const std::vector<std::pair<Species, unsigned int>> reactants, products;
       const Real reaction_rate;

       /// Adds the right hand side from this chemical reaction to the ODE.
       void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                    Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

       /// Adds the Jacobian matrix from this chemical reaction to the Jacobian of the ODE for the ODE solver.
       void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                         Matrix &J) override;

       /// Allocates space in the Jacobian for this chemical reaction (for sparse matrices).
       void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

       /// Adds the number of nonzero elements in the Jacobian this chemical reaction has in order to allocate enough space (for sparse matrices).
       void update_num_nonzero(unsigned int &num_nonzero) override;
     };



     template<typename Real, typename Matrix>
     ChemicalReaction<Real, Matrix>::ChemicalReaction(
         const std::vector<std::pair<Species, unsigned int>> reactants,
         const std::vector<std::pair<Species, unsigned int>> products,
         const Real reaction_rate)
         : reactants(reactants), products(products), reaction_rate(reaction_rate)
     {}



     template<typename Real, typename Matrix>
     void
     ChemicalReaction<Real, Matrix>::add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                                             Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
     {
       // From the law of mass action, each Species gets a contribution proportional to: k* \prod[ R_j^cr_j ]
       Real reaction_term = reaction_rate;
       for (const auto & chemical : reactants)
       {
         auto concentration = x(chemical.first.index);
         auto exponent = chemical.second;
         reaction_term *= std::pow(concentration, exponent);
       }

       // Each reactant loses at a rate based on reaction_term and the coefficient of that species.
       for (const auto & chemical : reactants)
       {
         rhs(chemical.first.index) -= reaction_term * chemical.second;
       }

       // Each product gains at a rate based on reaction_term and the coefficient of that species.
       for (const auto & chemical : products)
       {
         rhs(chemical.first.index) += reaction_term * chemical.second;
       }
     }



     template<typename Real, typename Matrix>
     void
     ChemicalReaction<Real, Matrix>::add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                                                  Matrix &J)
     {
       // The derivative of each component of the ODE is nonzero only with respect to each reactant.
       for (unsigned int i=0; i<reactants.size(); ++i)
       {
         // Compute the derivative with respect to reactant[i]
         Real deriv = reaction_rate;
         for (unsigned int j=0; j<reactants.size(); ++j)
         {
           auto concentration = x(reactants[j].first.index);
           auto coefficient = reactants[j].second;
           if (i==j)
           {
             // Derivative of x^n gives n*x^{n-1}
             deriv *= coefficient;
             coefficient -= 1;
             if (coefficient > 0)
               deriv *= std::pow(concentration, coefficient);
           }
           else
           {
             deriv *= std::pow(concentration, coefficient);
           }
         }

         // Derivative is now calculated up to plus/minus and the coefficient for each reactant/product.

         // Column index in the Jacobian is based on the Species we're taking the derivative with respect to.
         auto col = reactants[i].first.index;
         assert(col <= x.size());
         assert(col <= J.rows());
         assert(col <= J.cols());
         assert(col >= 0);

         // Add to the Jacobian for each reactant.
         for (const auto & chemical : reactants)
         {
           auto row = chemical.first.index;
           assert(row <= x.size());
           assert(row <= J.rows());
           assert(row <= J.cols());
           assert(row >= 0);
           // Reactants have negative derivatives.
           J.coeffRef(row,col) -= chemical.second * deriv;
         }

         // Add to the Jacobian for each product.
         for (const auto & chemical : products)
         {
           auto row = chemical.first.index;
           assert(row <= x.size());
           assert(row <= J.rows());
           assert(row <= J.cols());
           assert(row >= 0);
           // Products have positive derivatives.
           J.coeffRef(row,col) += chemical.second * deriv;
         }
       }
     }



     template<typename Real, typename Matrix>
     void
     ChemicalReaction<Real, Matrix>::add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list)
     {
       // Nonzero column entries for each reactant
       for (const auto & chemical_deriv : reactants)
       {
         const auto col = chemical_deriv.first.index;
         assert(col >= 0);

         // Each row corresponding to a reactant has a nonzero
         for (const auto & chemical : reactants)
         {
           const auto row = chemical.first.index;
           assert(row >= 0);
           triplet_list.push_back( Eigen::Triplet<Real>(row, col) );
         }

         // Each row corresponding to a product has a nonzero
         for (const auto & chemical : products)
         {
           const auto row = chemical.first.index;
           assert(row >= 0);
           triplet_list.push_back( Eigen::Triplet<Real>(row, col) );
         }
       }
     }



     template<typename Real, typename Matrix>
     void
     ChemicalReaction<Real, Matrix>::update_num_nonzero(unsigned int &num_nonzero)
     {
       // Each reactant/product has a derivative for each reactant.
       num_nonzero += reactants.size() * (reactants.size() + products.size());
     }



     /**
      * ParticleGrowth represents the underlying ChemicalReactions a Particle undergoes during Growth.
      * The psuedo-elementary step represented is
      *     sum(Reactant_i) + B ->[a*k] B + sum(Product_j)
      * Since the Particle B encompasses many species, this is really the set of reactions
      *     sum(Reactant_i) + B_n ->[a_n*k] B_{n+m} + sum(Product_j)
      * where a_n is the growth kernel and m is how much the size of a particle increases during growth.
      * Reactants and Products are not necessary, but the Particles will automatically be added to the underlying chemical reactions.
      */
     template<typename Real, typename Matrix>
     class ParticleGrowth : public RightHandSideContribution<Real, Matrix>
     {
     public:
       ParticleGrowth(const Particle particle,
                      const Real reaction_rate,
                      const unsigned int growth_amount,
                      const unsigned int max_particle_size,
                      const std::function<Real(const unsigned int)> growth_kernel,
                      const std::vector<std::pair<Species, unsigned int>> reactants,
                      const std::vector<std::pair<Species, unsigned int>> products);

       const Particle particle;
       const Real reaction_rate;
       const unsigned int growth_amount;
       const unsigned int max_particle_size;
       const std::function<Real(const unsigned int)> growth_kernel;
       const std::vector<std::pair<Species, unsigned int>> reactants, products;

       /// Adds the right hand side from this chemical reaction to the ODE.
       void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                    Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

       /// Adds the Jacobian matrix from this chemical reaction to the Jacobian of the ODE for the ODE solver.
       void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                         Matrix &J) override;

       /// Allocates space in the Jacobian for this chemical reaction (for sparse matrices).
       void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

       /// Adds the number of nonzero elements in the Jacobian this chemical reaction has in order to allocate enough space (for sparse matrices).
       void update_num_nonzero(unsigned int &num_nonzero) override;
     };



     template<typename Real, typename Matrix>
     ParticleGrowth<Real, Matrix>::ParticleGrowth(
         const Particle particle,
         const Real reaction_rate,
         const unsigned int growth_amount,
         const unsigned int max_particle_size,
         const std::function<Real(const unsigned int)> growth_kernel,
         const std::vector<std::pair<Species, unsigned int>> reactants,
         const std::vector<std::pair<Species, unsigned int>> products)
         : particle(particle),
           reaction_rate(reaction_rate),
           growth_amount(growth_amount),
           max_particle_size(max_particle_size),
           growth_kernel(growth_kernel),
           reactants(reactants),
           products(products)
         {}



     template<typename Real, typename Matrix>
     void
     ParticleGrowth<Real, Matrix>::add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                                           Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
     {
       for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
       {
         // Particle to grow
         const auto growth_particle = particle.get_particle_species(i);
         auto growth_reactants = reactants;
         growth_reactants.push_back({growth_particle, 1});

         // Particle created through growth
         const auto created_particle = particle.get_particle_species(i+growth_amount);
         auto growth_products = products;
         const unsigned int particle_size = (i - particle.index_start) + particle.first_size;
         const unsigned int created_particle_size = particle_size + growth_amount;
         if (created_particle_size <= max_particle_size)
         {
           // Only add a product if we care to track it.
           growth_products.push_back({created_particle, 1});
         }

         // Create the chemical reaction and add to the right hand side.
         ChemicalReaction<Real, Matrix> growth_rxn(growth_reactants,
                                                   growth_products,
                                                   reaction_rate * growth_kernel(particle_size));
         growth_rxn.add_contribution_to_rhs(x, rhs);
       }
     }



     template<typename Real, typename Matrix>
     void
     ParticleGrowth<Real, Matrix>::add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                                                Matrix &J)
     {
       for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
       {
         // Particle to grow
         const auto growth_particle = particle.get_particle_species(i);
         auto growth_reactants = reactants;
         growth_reactants.push_back({growth_particle, 1});

         // Particle created through growth
         const auto created_particle = particle.get_particle_species(i+growth_amount);
         auto growth_products = products;
         const unsigned int particle_size = (i - particle.index_start) + particle.first_size;
         const unsigned int created_particle_size = particle_size + growth_amount;
         if (created_particle_size <= max_particle_size)
         {
           // Only add a product if we care to track it.
           growth_products.push_back({created_particle, 1});
         }

         // Create the chemical reaction and add to the right hand side.
         ChemicalReaction<Real, Matrix> growth_rxn(growth_reactants,
                                                   growth_products,
                                                   reaction_rate * growth_kernel(particle_size));
         growth_rxn.add_contribution_to_jacobian(x,J);
       }
     }



    template<typename Real, typename Matrix>
    void
    ParticleGrowth<Real, Matrix>::add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list)
    {
      for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
      {
        // Particle to grow
        const auto growth_particle = particle.get_particle_species(i);
        auto growth_reactants = reactants;
        growth_reactants.push_back({growth_particle, 1});

        // Particle created through growth
        const auto created_particle = particle.get_particle_species(i+growth_amount);
        auto growth_products = products;
        const unsigned int particle_size = (i - particle.index_start) + particle.first_size;
        const unsigned int created_particle_size = particle_size + growth_amount;
        if (created_particle_size <= max_particle_size )
        {
        // So only add a product if we care to track it.
        growth_products.push_back({created_particle, 1});
        }

        // Create the chemical reaction and add to the right hand side.
        ChemicalReaction<Real, Matrix> growth_rxn(growth_reactants,
                                                  growth_products,
                                                  reaction_rate * growth_kernel(particle_size));
        growth_rxn.add_nonzero_to_jacobian(triplet_list);
      }
    }



    template<typename Real, typename Matrix>
    void
    ParticleGrowth<Real, Matrix>::update_num_nonzero(unsigned int &num_nonzero)
    {
      for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
      {
        // Particle to grow
        const auto growth_particle = particle.get_particle_species(i);
        auto growth_reactants = reactants;
        growth_reactants.push_back({growth_particle, 1});

        // Particle created through growth
        const auto created_particle = particle.get_particle_species(i+growth_amount);
        auto growth_products = products;
        const unsigned int particle_size = (i - particle.index_start) + particle.first_size;
        const unsigned int created_particle_size = particle_size + growth_amount;
        if (created_particle_size <= max_particle_size )
        {
          // So only add a product if we care to track it.
          growth_products.push_back({created_particle, 1});
        }

        // Create the chemical reaction and add to the right hand side.
        ChemicalReaction<Real, Matrix> growth_rxn(growth_reactants,
                                                  growth_products,
                                                  reaction_rate * growth_kernel(particle_size));
        // This is an overestimation, but that should not be a huge concern since it's basically just finding enough
        // space on the heap to store the matrix contiguously. So overestimation is fine where underestimation is suboptimal.
        growth_rxn.update_num_nonzero(num_nonzero);
      }
    }



    /**
     * ParticleAgglomeration represents the process of two particles sticking to each other to form a larger particle.
     * Specifically, the pseudo-elementary step
     *      sum(reactants) + A + B -> C + sum(products)
     * where A and B are particles of potentially different size categories and C is the resultant particle.
     * This leads to the reactions
     *      sum(reactants) + A_i + B_j ->[a_ij * k] C_ij + sum(products)
     */
    template<typename Real, typename Matrix>
    class ParticleAgglomeration : public RightHandSideContribution<Real, Matrix>
    {
    public:
      ParticleAgglomeration(const Particle particleA,
                            const Particle particleB,
                            const Real reaction_rate,
                            const unsigned int max_particle_size,
                            const std::function<Real(const unsigned int)> growth_kernel,
                            const std::vector<std::pair<Species, unsigned int>> reactants,
                            const std::vector<std::pair<Species, unsigned int>> products);

      const Particle particleA, particleB;
      const Real reaction_rate;
      const unsigned int max_particle_size;
      const std::function<Real(const unsigned int)> growth_kernel;
      const std::vector<std::pair<Species, unsigned int>> reactants, products;

      /// Adds the right hand side from this chemical reaction to the ODE.
      void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                   Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

      /// Adds the Jacobian matrix from this chemical reaction to the Jacobian of the ODE for the ODE solver.
      void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                        Matrix &J) override;

      /// Allocates space in the Jacobian for this chemical reaction (for sparse matrices).
      void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

      /// Adds the number of nonzero elements in the Jacobian this chemical reaction has in order to allocate enough space (for sparse matrices).
      void update_num_nonzero(unsigned int &num_nonzero) override;
    };




    template<typename Real, typename Matrix>
    ParticleAgglomeration<Real, Matrix>::ParticleAgglomeration(
        const Particle particleA,
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



    template<typename Real, typename Matrix>
    void
    ParticleAgglomeration<Real, Matrix>::add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
    {
      // Loop through every particleA
      for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
      {
        // Define Species of particleA
        const auto agglomA = particleA.get_particle_species(indexA);
        const unsigned int A_size = (indexA - particleA.index_start) + particleA.first_size;
        auto agglom_reactants = reactants;
        agglom_reactants.push_back({agglomA,1});

        // Loop through every particleB to let the particleA species react with all particleB species.
        // If there's overlap in the particles, we don't want to do double count. i.e. we don't want to include
        // both of
        //    Particle_3 + Particle_4 -> Particle_7
        //    Particle_4 + Particle_3 -> Particle_7
        const unsigned int corrected_index = std::max(particleB.index_start, indexA);
        for (unsigned int indexB=corrected_index; indexB <= particleB.index_end; ++indexB)
        {
         // Define Species of particleB
         const auto agglomB = particleB.get_particle_species(indexB);
         const unsigned int B_size = (indexB - particleB.index_start) + particleB.first_size;
         agglom_reactants.push_back({agglomB,1});

         // Define resultant particle
         const unsigned int created_size = A_size + B_size;
         const unsigned int created_index = (created_size - particleA.first_size) + particleA.index_start;
         const Species created_particle(created_index);
         // If the created particle is larger than the largest tracked particle, then don't add it as a product.
         // This can be checked by seeing if created_index is larger than the size of the vector.
         auto agglom_products = products;
         if (created_index < x.size())
         {
           agglom_products.push_back({created_particle,1});
         }

         // Define chemical reaction and add to the right hand side
         const auto k = reaction_rate * growth_kernel(A_size) * growth_kernel(B_size);
         ChemicalReaction<Real, Matrix> agglom_rxn(agglom_reactants, agglom_products, k);
         agglom_rxn.add_contribution_to_rhs(x, rhs);
         agglom_reactants.pop_back(); // Last reactant needs to be replaced in next iteration of loop.
        }
      }
    }



    template<typename Real, typename Matrix>
    void
    ParticleAgglomeration<Real, Matrix>::add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                                                      Matrix &J)
    {
      // Loop through every particleA
      for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
      {
        // Define Species of particleA
        const auto agglomA = particleA.get_particle_species(indexA);
        const unsigned int A_size = (indexA - particleA.index_start) + particleA.first_size;
        auto agglom_reactants = reactants;
        agglom_reactants.push_back({agglomA,1});

        // Loop through every particleB to let the particleA species react with all particleB species.
        // If there's overlap in the particles, we don't want to do double count. i.e. we don't want to include
        // both of
        //    Particle_3 + Particle_4 -> Particle_7
        //    Particle_4 + Particle_3 -> Particle_7
        const unsigned int corrected_index = std::max(particleB.index_start, indexA);
        for (unsigned int indexB=corrected_index; indexB <= particleB.index_end; ++indexB)
        {
          // Define Species of particleB
          const auto agglomB = particleB.get_particle_species(indexB);
          const unsigned int B_size = (indexB - particleB.index_start) + particleB.first_size;
          agglom_reactants.push_back({agglomB,1});

          // Define resultant particle
          const unsigned int created_size = A_size + B_size;
          const unsigned int created_index = (created_size - particleA.first_size) + particleA.index_start;
          const Species created_particle(created_index);
          // If the created particle is larger than the largest tracked particle, then don't add it as a product.
          // This can be checked by seeing if created_index is larger than the size of the vector.
          auto agglom_products = products;
          if (created_index < x.size())
          {
            agglom_products.push_back({created_particle,1});
          }

          // Define chemical reaction and add to the right hand side
          const auto k = reaction_rate * growth_kernel(A_size) * growth_kernel(B_size);
          ChemicalReaction<Real, Matrix> agglom_rxn(agglom_reactants, agglom_products, k);
          agglom_rxn.add_contribution_to_jacobian(x, J);
          agglom_reactants.pop_back(); // Last reactant needs to be replaced in next iteration of loop.
        }
      }
    }



    template<typename Real, typename Matrix>
    void
    ParticleAgglomeration<Real, Matrix>::add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list)
    {
      // Loop through every particleA
      for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
      {
        // Define Species of particleA
        const auto agglomA = particleA.get_particle_species(indexA);
        const unsigned int A_size = (indexA - particleA.index_start) + particleA.first_size;
        auto agglom_reactants = reactants;
        agglom_reactants.push_back({agglomA,1});

        // Loop through every particleB to let the particleA species react with all particleB species.
        // If there's overlap in the particles, we don't want to do double count. i.e. we don't want to include
        // both of
        //    Particle_3 + Particle_4 -> Particle_7
        //    Particle_4 + Particle_3 -> Particle_7
        const unsigned int corrected_index = std::max(particleB.index_start, indexA);
        for (unsigned int indexB=corrected_index; indexB <= particleB.index_end; ++indexB)
        {
          // Define Species of particleB
          const auto agglomB = particleB.get_particle_species(indexB);
          const unsigned int B_size = (indexB - particleB.index_start) + particleB.first_size;
          agglom_reactants.push_back({agglomB,1});

          // Define resultant particle
          const unsigned int created_size = A_size + B_size;
          const unsigned int created_index = (created_size - particleA.first_size) + particleA.index_start;
          const Species created_particle(created_index);
          // If the created particle is larger than the largest tracked particle, then don't add it as a product.
          // This can be checked by seeing if created_index is larger than the size of the vector.
          auto agglom_products = products;
          if (created_size <= max_particle_size)
          {
            agglom_products.push_back({created_particle,1});
          }

          // Define chemical reaction and add to the right hand side
          const auto k = reaction_rate * growth_kernel(A_size) * growth_kernel(B_size);
          ChemicalReaction<Real, Matrix> agglom_rxn(agglom_reactants, agglom_products, k);
          agglom_rxn.add_nonzero_to_jacobian(triplet_list);
          agglom_reactants.pop_back(); // Last reactant needs to be replaced in next iteration of loop.
        }
      }
    }



    template<typename Real, typename Matrix>
    void
    ParticleAgglomeration<Real, Matrix>::update_num_nonzero(unsigned int &num_nonzero)
    {
      // Loop through every particleA
      for (unsigned int indexA=particleA.index_start; indexA <= particleA.index_end; ++indexA)
      {
        // Define Species of particleA
        const auto agglomA = particleA.get_particle_species(indexA);
        const unsigned int A_size = (indexA - particleA.index_start) + particleA.first_size;
        auto agglom_reactants = reactants;
        agglom_reactants.push_back({agglomA,1});

        // Loop through every particleB to let the particleA species react with all particleB species.
        // If there's overlap in the particles, we don't want to do double count. i.e. we don't want to include
        // both of
        //    Particle_3 + Particle_4 -> Particle_7
        //    Particle_4 + Particle_3 -> Particle_7
        const unsigned int corrected_index = std::max(particleB.index_start, indexA);
        for (unsigned int indexB=corrected_index; indexB <= particleB.index_end; ++indexB)
        {
          // Define Species of particleB
          const auto agglomB = particleB.get_particle_species(indexB);
          const unsigned int B_size = (indexB - particleB.index_start) + particleB.first_size;
          agglom_reactants.push_back({agglomB,1});

          // Define resultant particle
          const unsigned int created_size = A_size + B_size;
          const unsigned int created_index = (created_size - particleA.first_size) + particleA.index_start;
          const Species created_particle(created_index);
          // If the created particle is larger than the largest tracked particle, then don't add it as a product.
          // This can be checked by seeing if created_index is larger than the size of the vector.
          auto agglom_products = products;
          if (created_size <= max_particle_size)
          {
            agglom_products.push_back({created_particle,1});
          }

          // Define chemical reaction and add to the right hand side
          const auto k = reaction_rate * growth_kernel(A_size) * growth_kernel(B_size);
          ChemicalReaction<Real, Matrix> agglom_rxn(agglom_reactants, agglom_products, k);
          agglom_rxn.update_num_nonzero(num_nonzero);
          agglom_reactants.pop_back(); // Last reactant needs to be replaced in next iteration of loop.
        }
      }
    }
}










#endif //MEPBM_CHEMICAL_REACTION_H
