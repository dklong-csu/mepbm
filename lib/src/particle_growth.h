#ifndef MEPBM_PARTICLE_GROWTH_H
#define MEPBM_PARTICLE_GROWTH_H


#include "chemical_reaction.h"
#include "particle.h"
#include "species.h"


namespace MEPBM {
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
  class ParticleGrowth
  {
  public:
    ParticleGrowth(const Particle particle,
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

    const Particle particle;
    const Real reaction_rate;
    const unsigned int growth_amount;
    const unsigned int max_particle_size;
    const std::function<Real(const unsigned int)> growth_kernel;
    const std::vector<std::pair<Species, unsigned int>> reactants, products;

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void*)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void * user_data)
      {
        int output_value;
        // For each particle, compose the chemical reaction and then apply it.
        for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
        {
          // Extract particle species
          auto reaction_particle = particle.species(i);
          // Form the complete list of reactants
          auto all_reactants = reactants;
          all_reactants.push_back({reaction_particle, 1}); // 1 particle involved as a reactant

          // Form the complete list of products
          auto all_products = products;
          // If the created particle is a tracked size, then add it to the product list
          if (particle.size(i) + growth_amount <= max_particle_size)
          {
            // Particle is indexed in increments of 1 so the growth amount of the particle also gives the index offset.
            auto product_particle = particle.species(i+growth_amount);
            all_products.push_back({product_particle,1}); // 1 particle involved as a product
          }

          // Form a chemical reaction
          // Growth kernel times the base reaction rate gives the overall reaction rate
          const Real rate = reaction_rate * growth_kernel(particle.size(i));
          ChemicalReaction<Real, Matrix> rxn(all_reactants, all_products, rate);

          // Get the right hand side function
          auto rhs = rxn.rhs_function();

          // Apply the right hand side function
          // FIXME: should this be += instead of =?
          output_value = rhs(time, x, x_dot, user_data);
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
        // For each particle, compose the chemical reaction and then apply it.
        for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
        {
          // Extract particle species
          auto reaction_particle = particle.species(i);
          // Form the complete list of reactants
          auto all_reactants = reactants;
          all_reactants.push_back({reaction_particle, 1}); // 1 particle involved as a reactant

          // Form the complete list of products
          auto all_products = products;
          // If the created particle is a tracked size, then add it to the product list
          if (particle.size(i) + growth_amount <= max_particle_size)
          {
            // Particle is indexed in increments of 1 so the growth amount of the particle also gives the index offset.
            auto product_particle = particle.species(i+growth_amount);
            all_products.push_back({product_particle,1}); // 1 particle involved as a product
          }

          // Form a chemical reaction
          // Growth kernel times the base reaction rate gives the overall reaction rate
          const Real rate = reaction_rate * growth_kernel(particle.size(i));
          ChemicalReaction<Real, Matrix> rxn(all_reactants, all_products, rate);

          // Get the Jacobian function
          auto jac = rxn.jacobian_function();

          // Apply the right hand side function
          // FIXME: should this be += instead of =?
          output_value = jac(time, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
        }
        return output_value;
      };
      return fcn;
    }
  };



  /**
      * ParticleGrowth represents the underlying ChemicalReactions a Particle undergoes during Growth.
      * The psuedo-elementary step represented is
      *     sum(Reactant_i) + B ->[a*k] B + sum(Product_j)
      * Since the Particle B encompasses many species, this is really the set of reactions
      *     sum(Reactant_i) + B_n ->[a_n*k] B_{n+m} + sum(Product_j)
      * where a_n is the growth kernel and m is how much the size of a particle increases during growth.
      * Reactants and Products are not necessary, but the Particles will automatically be added to the underlying chemical reactions.
      * This is a partial specialization for sparse matrices using compressed column storage. The formation of the
      * Jacobian needs to be handled differently than in the dense matrix case.
      */
  template<typename Real>
  class ParticleGrowth<Real, Eigen::SparseMatrix<Real>>
  {
  public:
    ParticleGrowth(const Particle particle,
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

    const Particle particle;
    const Real reaction_rate;
    const unsigned int growth_amount;
    const unsigned int max_particle_size;
    const std::function<Real(const unsigned int)> growth_kernel;
    const std::vector<std::pair<Species, unsigned int>> reactants, products;

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void*)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void * user_data)
      {
        int output_value;
        // For each particle, compose the chemical reaction and then apply it.
        for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
        {
          // Extract particle species
          auto reaction_particle = particle.species(i);
          // Form the complete list of reactants
          auto all_reactants = reactants;
          all_reactants.push_back({reaction_particle, 1}); // 1 particle involved as a reactant

          // Form the complete list of products
          auto all_products = products;
          // If the created particle is a tracked size, then add it to the product list
          if (particle.size(i) + growth_amount <= max_particle_size)
          {
            // Particle is indexed in increments of 1 so the growth amount of the particle also gives the index offset.
            auto product_particle = particle.species(i+growth_amount);
            all_products.push_back({product_particle,1}); // 1 particle involved as a product
          }

          // Form a chemical reaction
          // Growth kernel times the base reaction rate gives the overall reaction rate
          const Real rate = reaction_rate * growth_kernel(particle.size(i));
          ChemicalReaction<Real, Eigen::SparseMatrix<Real>> rxn(all_reactants, all_products, rate);

          // Get the right hand side function
          auto rhs = rxn.rhs_function();

          // Apply the right hand side function
          // FIXME: should this be += instead of =?
          output_value = rhs(time, x, x_dot, user_data);
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
        // For each particle, compose the chemical reaction and then apply it.
        for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
        {
          // Extract particle species
          auto reaction_particle = particle.species(i);
          // Form the complete list of reactants
          auto all_reactants = reactants;
          all_reactants.push_back({reaction_particle, 1}); // 1 particle involved as a reactant

          // Form the complete list of products
          auto all_products = products;
          // If the created particle is a tracked size, then add it to the product list
          if (particle.size(i) + growth_amount <= max_particle_size)
          {
            // Particle is indexed in increments of 1 so the growth amount of the particle also gives the index offset.
            auto product_particle = particle.species(i+growth_amount);
            all_products.push_back({product_particle,1}); // 1 particle involved as a product
          }

          // Form a chemical reaction
          // Growth kernel times the base reaction rate gives the overall reaction rate
          const Real rate = reaction_rate * growth_kernel(particle.size(i));
          ChemicalReaction<Real, Eigen::SparseMatrix<Real>> rxn(all_reactants, all_products, rate);

          // Get the Jacobian function
          auto jac = rxn.jacobian_function();

          // Apply the right hand side function
          // FIXME: should this be += instead of =?
          output_value = jac(x, triplet_list, Jacobian);
        }
        return output_value;
      };
      return fcn;
    }
  };



  /**
      * ParticleGrowth represents the underlying ChemicalReactions a Particle undergoes during Growth.
      * The psuedo-elementary step represented is
      *     sum(Reactant_i) + B ->[a*k] B + sum(Product_j)
      * Since the Particle B encompasses many species, this is really the set of reactions
      *     sum(Reactant_i) + B_n ->[a_n*k] B_{n+m} + sum(Product_j)
      * where a_n is the growth kernel and m is how much the size of a particle increases during growth.
      * Reactants and Products are not necessary, but the Particles will automatically be added to the underlying chemical reactions.
      * This is a partial specialization for sparse matrices using compressed row storage. The formation of the
      * Jacobian needs to be handled differently than in the dense matrix case.
      */
  template<typename Real>
  class ParticleGrowth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
  {
  public:
    ParticleGrowth(const Particle particle,
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

    const Particle particle;
    const Real reaction_rate;
    const unsigned int growth_amount;
    const unsigned int max_particle_size;
    const std::function<Real(const unsigned int)> growth_kernel;
    const std::vector<std::pair<Species, unsigned int>> reactants, products;

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void*)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void * user_data)
      {
        int output_value;
        // For each particle, compose the chemical reaction and then apply it.
        for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
        {
          // Extract particle species
          auto reaction_particle = particle.species(i);
          // Form the complete list of reactants
          auto all_reactants = reactants;
          all_reactants.push_back({reaction_particle, 1}); // 1 particle involved as a reactant

          // Form the complete list of products
          auto all_products = products;
          // If the created particle is a tracked size, then add it to the product list
          if (particle.size(i) + growth_amount <= max_particle_size)
          {
            // Particle is indexed in increments of 1 so the growth amount of the particle also gives the index offset.
            // FIXME: use index() function to be more clear?
            auto product_particle = particle.species(i+growth_amount);
            all_products.push_back({product_particle,1}); // 1 particle involved as a product
          }

          // Form a chemical reaction
          // Growth kernel times the base reaction rate gives the overall reaction rate
          const Real rate = reaction_rate * growth_kernel(particle.size(i));
          ChemicalReaction<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> rxn(all_reactants, all_products, rate);

          // Get the right hand side function
          auto rhs = rxn.rhs_function();

          // Apply the right hand side function
          // FIXME: should this be += instead of =?
          output_value = rhs(time, x, x_dot, user_data);
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
        // For each particle, compose the chemical reaction and then apply it.
        for (unsigned int i=particle.index_start; i<=particle.index_end; ++i)
        {
          // Extract particle species
          auto reaction_particle = particle.species(i);
          // Form the complete list of reactants
          auto all_reactants = reactants;
          all_reactants.push_back({reaction_particle, 1}); // 1 particle involved as a reactant

          // Form the complete list of products
          auto all_products = products;
          // If the created particle is a tracked size, then add it to the product list
          if (particle.size(i) + growth_amount <= max_particle_size)
          {
            // Particle is indexed in increments of 1 so the growth amount of the particle also gives the index offset.
            auto product_particle = particle.species(i+growth_amount);
            all_products.push_back({product_particle,1}); // 1 particle involved as a product
          }

          // Form a chemical reaction
          // Growth kernel times the base reaction rate gives the overall reaction rate
          const Real rate = reaction_rate * growth_kernel(particle.size(i));
          ChemicalReaction<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> rxn(all_reactants, all_products, rate);

          // Get the Jacobian function
          auto jac = rxn.jacobian_function();

          // Apply the right hand side function
          // FIXME: should this be += instead of =?
          output_value = jac(x, triplet_list, Jacobian);
        }
        return output_value;
      };
      return fcn;
    }
  };
}
#endif //MEPBM_PARTICLE_GROWTH_H
