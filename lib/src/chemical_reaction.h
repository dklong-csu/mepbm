#ifndef MEPBM_CHEMICAL_REACTION_H
#define MEPBM_CHEMICAL_REACTION_H

#include "particle.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>
#include <utility>
#include <functional>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_matrix.h>
#include <cassert>




namespace MEPBM {
  /**
   * An object that contains instructions for the right-hand side and Jacobian of the set of
   * ordinary differentials equations that arise from modeling a chemical reaction with the
   * law of mass action.
   */
  template<typename Real, typename Matrix>
  class ChemicalReaction {
  public:
    /// Constructor.
    ChemicalReaction(const std::vector< std::pair<Species, unsigned int> > reactants,
                     const std::vector< std::pair<Species, unsigned int> > products,
                     const Real rate)
    : reactants(reactants),
      products(products),
      rate(rate)
    {}

    const std::vector< std::pair<Species, unsigned int> > reactants; /// The species and multiplicity of each reactant.
    const std::vector< std::pair<Species, unsigned int> > products;  /// The species and multiplicity of each product.
    const Real rate; /// The reaction rate.

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void*)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void * user_data)
      {
        // We always use N_Vectors based on the Eigen library, so use that to work with the vector.
        auto x_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);
        auto x_dot_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x_dot->content);

        // The right-hand side contribution from this particular reaction is based on the law of mass action.
        //    product( reactant_i ^ multiplicity_i ) * rate
        // Then involved reactant and product is scaled by its multiplicity.
        Real rxn = rate;
        for (const auto & r : reactants)
        {
          auto index = r.first.index;
          assert(index >= 0); // make sure nothing is weird with the index
          assert(index < x_vec->size()); // make sure nothing is weird with the index
          assert(index < x_dot_vec->size()); // make sure nothing is weird with the index
          auto exponent = r.second;
          auto concentration = (*x_vec)(index);
          rxn *= std::pow(concentration, exponent);
        }

        // Now distribute the reaction term to all relevant locations in the vector.
        for (const auto & r : reactants)
        {
          auto index = r.first.index;
          assert(index >= 0); // make sure nothing is weird with the index
          assert(index < x_vec->size()); // make sure nothing is weird with the index
          assert(index < x_dot_vec->size()); // make sure nothing is weird with the index
          auto multiplicity = r.second;
          (*x_dot_vec)(index) -= multiplicity * rxn; // -= because reactant is being removed
        }

        for (const auto & p : products)
        {
          auto index = p.first.index;
          assert(index >= 0); // make sure nothing is weird with the index
          assert(index < x_vec->size()); // make sure nothing is weird with the index
          assert(index < x_dot_vec->size()); // make sure nothing is weird with the index
          auto multiplicity = p.second;
          (*x_dot_vec)(index) += multiplicity * rxn; // += because product is being added
        }
        int success = 0;
        return success;
      };
      return fcn;
    }

    /// Returns a function for the Jacobian of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, SUNMatrix, void *, N_Vector, N_Vector, N_Vector)>
    jacobian_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, SUNMatrix Jacobian, void * user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
      {
        // Pull out the underlying matrix from the SUNMatrix
        auto J = static_cast<Matrix*>(Jacobian->content);
        // Pull out the underlying state vector from the N_Vector
        auto x_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);

        // The derivative of the rhs is nonzero with respect to each reactant.
        for (unsigned int i=0; i<reactants.size(); ++i)
        {
          // Compute the derivative with respect to reactant[i]
          Real deriv = rate;
          for (unsigned int j=0; j<reactants.size(); ++j)
          {
            auto concentration = (*x_vec)(reactants[j].first.index);
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
          assert(col <= x_vec->size());
          assert(col <= J->rows());
          assert(col <= J->cols());
          assert(col >= 0);

          // Add to the Jacobian for each reactant.
          for (const auto & chemical : reactants)
          {
            auto row = chemical.first.index;
            assert(row <= x_vec->size());
            assert(row <= J->rows());
            assert(row <= J->cols());
            assert(row >= 0);
            // Reactants have negative derivatives.
            J->coeffRef(row,col) -= chemical.second * deriv;
          }

          // Add to the Jacobian for each product.
          for (const auto & chemical : products)
          {
            auto row = chemical.first.index;
            assert(row <= x_vec->size());
            assert(row <= J->rows());
            assert(row <= J->cols());
            assert(row >= 0);
            // Products have positive derivatives.
            J->coeffRef(row,col) += chemical.second * deriv;
          }
        }

        int success = 0;
        return success;
      };
      return fcn;
    }

  };



  /**
   * An object that contains instructions for the right-hand side and Jacobian of the set of
   * ordinary differentials equations that arise from modeling a chemical reaction with the
   * law of mass action. If using a sparse matrix, the formation of the Jacobian needs to be
   * handled differently to achieve performance, so a partial specialization is used. This
   * specialization is for compressed column storage.
   */
  template<typename Real>
  class ChemicalReaction<Real, Eigen::SparseMatrix<Real>> {
  public:
    /// Constructor.
    ChemicalReaction(const std::vector< std::pair<Species, unsigned int> > reactants,
                     const std::vector< std::pair<Species, unsigned int> > products,
                     const Real rate)
        : reactants(reactants),
          products(products),
          rate(rate)
    {}

    const std::vector< std::pair<Species, unsigned int> > reactants; /// The species and multiplicity of each reactant.
    const std::vector< std::pair<Species, unsigned int> > products;  /// The species and multiplicity of each product.
    const Real rate; /// The reaction rate.

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void*)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void * user_data)
      {
        // We always use N_Vectors based on the Eigen library, so use that to work with the vector.
        auto x_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);
        auto x_dot_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x_dot->content);

        // The right-hand side contribution from this particular reaction is based on the law of mass action.
        //    product( reactant_i ^ multiplicity_i ) * rate
        // Then involved reactant and product is scaled by its multiplicity.
        Real rxn = rate;
        for (const auto & r : reactants)
        {
          auto index = r.first.index;
          assert(index >= 0); // make sure nothing is weird with the index
          assert(index < x_vec->size()); // make sure nothing is weird with the index
          assert(index < x_dot_vec->size()); // make sure nothing is weird with the index
          auto exponent = r.second;
          auto concentration = (*x_vec)(index);
          rxn *= std::pow(concentration, exponent);
        }

        // Now distribute the reaction term to all relevant locations in the vector.
        for (const auto & r : reactants)
        {
          auto index = r.first.index;
          assert(index >= 0); // make sure nothing is weird with the index
          assert(index < x_vec->size()); // make sure nothing is weird with the index
          assert(index < x_dot_vec->size()); // make sure nothing is weird with the index
          auto multiplicity = r.second;
          (*x_dot_vec)(index) -= multiplicity * rxn; // -= because reactant is being removed
        }

        for (const auto & p : products)
        {
          auto index = p.first.index;
          assert(index >= 0); // make sure nothing is weird with the index
          assert(index < x_vec->size()); // make sure nothing is weird with the index
          assert(index < x_dot_vec->size()); // make sure nothing is weird with the index
          auto multiplicity = p.second;
          (*x_dot_vec)(index) += multiplicity * rxn; // += because product is being added
        }
        int success = 0;
        return success;
      };
      return fcn;
    }

    /**
     * Returns a function for the Jacobian of the ODE. This function is not yet compatible with the SUNDIALS API.
     * Sparse matrices in Eigen are best created from a triplet list, which must be done in one step. So instead
     * of directly adding to the Jacobian matrix, this function appends to the provided triplet list and must be
     * converted to a sparse matrix is a wrapper function for SUNDIALS. The Jacobian is provided to this function
     * strictly to check indices.
     */
    std::function<int(N_Vector, std::vector<Eigen::Triplet<Real>> &, SUNMatrix)>
    jacobian_function() const {
      auto fcn = [&](N_Vector x, std::vector<Eigen::Triplet<Real>> &triplet_list, SUNMatrix Jacobian)
      {
        // Pull out the underlying matrix from the SUNMatrix
        auto J = static_cast<Eigen::SparseMatrix<Real>*>(Jacobian->content);
        // Pull out the underlying state vector from the N_Vector
        auto x_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);

        // The derivative of the rhs is nonzero with respect to each reactant.
        for (unsigned int i=0; i<reactants.size(); ++i)
        {
          // Compute the derivative with respect to reactant[i]
          Real deriv = rate;
          for (unsigned int j=0; j<reactants.size(); ++j)
          {
            auto concentration = (*x_vec)(reactants[j].first.index);
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
          assert(col <= x_vec->size());
          assert(col <= J->rows());
          assert(col <= J->cols());
          assert(col >= 0);

          // Add to the triplet list for each reactant.
          for (const auto & chemical : reactants)
          {
            auto row = chemical.first.index;
            assert(row <= x_vec->size());
            assert(row <= J->rows());
            assert(row <= J->cols());
            assert(row >= 0);
            // Reactants have negative derivatives.
            auto rxn = -1.0 * chemical.second * deriv;
            if (rxn != 0.0)
              triplet_list.push_back(Eigen::Triplet<Real>(row, col, rxn));
          }

          // Add to the triplet list for each product.
          for (const auto & chemical : products)
          {
            auto row = chemical.first.index;
            assert(row <= x_vec->size());
            assert(row <= J->rows());
            assert(row <= J->cols());
            assert(row >= 0);
            // Products have positive derivatives.
            auto rxn = chemical.second * deriv;
            if (rxn != 0.0)
              triplet_list.push_back(Eigen::Triplet<Real>(row, col, rxn));
          }
        }

        int success = 0;
        return success;
      };
      return fcn;
    }

  };



  /**
   * An object that contains instructions for the right-hand side and Jacobian of the set of
   * ordinary differentials equations that arise from modeling a chemical reaction with the
   * law of mass action. If using a sparse matrix, the formation of the Jacobian needs to be
   * handled differently to achieve performance, so a partial specialization is used. This
   * specialization is for compressed row storage.
   */
  template<typename Real>
  class ChemicalReaction<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> {
  public:
    /// Constructor.
    ChemicalReaction(const std::vector< std::pair<Species, unsigned int> > reactants,
                     const std::vector< std::pair<Species, unsigned int> > products,
                     const Real rate)
        : reactants(reactants),
          products(products),
          rate(rate)
    {}

    const std::vector< std::pair<Species, unsigned int> > reactants; /// The species and multiplicity of each reactant.
    const std::vector< std::pair<Species, unsigned int> > products;  /// The species and multiplicity of each product.
    const Real rate; /// The reaction rate.

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void*)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void * user_data)
      {
        // We always use N_Vectors based on the Eigen library, so use that to work with the vector.
        auto x_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);
        auto x_dot_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x_dot->content);

        // The right-hand side contribution from this particular reaction is based on the law of mass action.
        //    product( reactant_i ^ multiplicity_i ) * rate
        // Then involved reactant and product is scaled by its multiplicity.
        Real rxn = rate;
        for (const auto & r : reactants)
        {
          auto index = r.first.index;
          assert(index >= 0); // make sure nothing is weird with the index
          assert(index < x_vec->size()); // make sure nothing is weird with the index
          assert(index < x_dot_vec->size()); // make sure nothing is weird with the index
          auto exponent = r.second;
          auto concentration = (*x_vec)(index);
          rxn *= std::pow(concentration, exponent);
        }

        // Now distribute the reaction term to all relevant locations in the vector.
        for (const auto & r : reactants)
        {
          auto index = r.first.index;
          assert(index >= 0); // make sure nothing is weird with the index
          assert(index < x_vec->size()); // make sure nothing is weird with the index
          assert(index < x_dot_vec->size()); // make sure nothing is weird with the index
          auto multiplicity = r.second;
          (*x_dot_vec)(index) -= multiplicity * rxn; // -= because reactant is being removed
        }

        for (const auto & p : products)
        {
          auto index = p.first.index;
          assert(index >= 0); // make sure nothing is weird with the index
          assert(index < x_vec->size()); // make sure nothing is weird with the index
          assert(index < x_dot_vec->size()); // make sure nothing is weird with the index
          auto multiplicity = p.second;
          (*x_dot_vec)(index) += multiplicity * rxn; // += because product is being added
        }
        int success = 0;
        return success;
      };
      return fcn;
    }

    /**
     * Returns a function for the Jacobian of the ODE. This function is not yet compatible with the SUNDIALS API.
     * Sparse matrices in Eigen are best created from a triplet list, which must be done in one step. So instead
     * of directly adding to the Jacobian matrix, this function appends to the provided triplet list and must be
     * converted to a sparse matrix is a wrapper function for SUNDIALS. The Jacobian is provided to this function
     * strictly to check indices.
     */
    std::function<int(N_Vector, std::vector<Eigen::Triplet<Real>>&, SUNMatrix)>
    jacobian_function() const {
      auto fcn = [&](N_Vector x, std::vector<Eigen::Triplet<Real>> &triplet_list, SUNMatrix Jacobian)
      {
        // Pull out the underlying matrix from the SUNMatrix
        auto J = static_cast<Eigen::SparseMatrix<Real, Eigen::RowMajor>*>(Jacobian->content);
        // Pull out the underlying state vector from the N_Vector
        auto x_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x->content);

        // The derivative of the rhs is nonzero with respect to each reactant.
        for (unsigned int i=0; i<reactants.size(); ++i)
        {
          // Compute the derivative with respect to reactant[i]
          Real deriv = rate;
          for (unsigned int j=0; j<reactants.size(); ++j)
          {
            auto concentration = (*x_vec)(reactants[j].first.index);
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
          assert(col <= x_vec->size());
          assert(col <= J->rows());
          assert(col <= J->cols());
          assert(col >= 0);

          // Add to the triplet list for each reactant.
          for (const auto & chemical : reactants)
          {
            auto row = chemical.first.index;
            assert(row <= x_vec->size());
            assert(row <= J->rows());
            assert(row <= J->cols());
            assert(row >= 0);
            // Reactants have negative derivatives.
            auto rxn = -1.0 * chemical.second * deriv;
            if (rxn != 0.0)
              triplet_list.push_back(Eigen::Triplet<Real>(row, col, rxn));
          }

          // Add to the triplet list for each product.
          for (const auto & chemical : products)
          {
            auto row = chemical.first.index;
            assert(row <= x_vec->size());
            assert(row <= J->rows());
            assert(row <= J->cols());
            assert(row >= 0);
            // Products have positive derivatives.
            auto rxn = chemical.second * deriv;
            if (rxn != 0.0)
              triplet_list.push_back(Eigen::Triplet<Real>(row, col, rxn));
          }
        }

        int success = 0;
        return success;
      };
      return fcn;
    }

  };
}

#endif //MEPBM_CHEMICAL_REACTION_H
