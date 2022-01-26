#ifndef MEPBM_CHEMICAL_REACTION_NETWORK_H
#define MEPBM_CHEMICAL_REACTION_NETWORK_H


#include "src/chemical_reaction.h"
#include "src/particle_agglomeration.h"
#include "src/particle_growth.h"
#include <vector>



namespace MEPBM {
  /**
   * This represents a complete system of chemical reactions that form a complete mechanism. The mechanism
   * includes a set of individual reactions, a set of growth processes, and a set of agglomeration processes.
   */
   template<typename Real, typename Matrix>
   class ChemicalReactionNetwork {
   public:
     /// Constructor
     ChemicalReactionNetwork(std::vector <MEPBM::ChemicalReaction<Real, Matrix>> reactions,
                             std::vector <MEPBM::ParticleGrowth<Real, Matrix>> growth_processes,
                             std::vector <MEPBM::ParticleAgglomeration<Real, Matrix>> agglomeration_processes)
         : reactions(reactions),
           growth_processes(growth_processes),
           agglomeration_processes(agglomeration_processes) {}

     /// All of the individual reactions in the mechanism
     std::vector <MEPBM::ChemicalReaction<Real, Matrix>> reactions;

     /// All of the growth processes in the mechanism
     std::vector <MEPBM::ParticleGrowth<Real, Matrix>> growth_processes;

     ///All of the agglomeration processes in the mechanism
     std::vector <MEPBM::ParticleAgglomeration<Real, Matrix>> agglomeration_processes;

     /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
     std::function<int(Real, N_Vector, N_Vector, void *)>
     rhs_function() const {
       auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void *user_data) {
         int output_value;

         // Make sure x_dot = 0 to begin with
         auto x_dot_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x_dot->content);
         x_dot_vec->setZero();

         // Loop through all individual reactions
         for (auto rxn: reactions) {
           auto rhs = rxn.rhs_function();
           output_value = rhs(time, x, x_dot, user_data);
         }

         // Loop through all growth processes
         for (auto growth: growth_processes) {
           auto rhs = growth.rhs_function();
           output_value = rhs(time, x, x_dot, user_data);
         }

         // Loop through all agglomeration processes
         for (auto agglomeration: agglomeration_processes) {
           auto rhs = agglomeration.rhs_function();
           output_value = rhs(time, x, x_dot, user_data);
         }
         return output_value;
       };
       return fcn;
     }


     /// Returns a function for the Jacobian of the ODE that is compatible with the SUNDIALS API.
     std::function<int(Real, N_Vector, N_Vector, SUNMatrix, void *, N_Vector, N_Vector, N_Vector)>
     jacobian_function() const {
       auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2,
                      N_Vector tmp3) {
         int output_value;

         // Make sure the Jacobian is zero to start
         auto J_mat = static_cast<Matrix*>(J->content);
         J_mat->setZero();

         // Loop through all individual reactions
         for (auto rxn: reactions) {
           auto jac = rxn.jacobian_function();
           output_value = jac(time, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
         }

         // Loop through all growth processes
         for (auto growth: growth_processes) {
           auto jac = growth.jacobian_function();
           output_value = jac(time, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
         }

         // Loop through all agglomeration processes
         for (auto agglomeration: agglomeration_processes) {
           auto jac = agglomeration.jacobian_function();
           output_value = jac(time, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
         }
         return output_value;
       };
       return fcn;
     }
   };



  /**
  * This represents a complete system of chemical reactions that form a complete mechanism. The mechanism
  * includes a set of individual reactions, a set of growth processes, and a set of agglomeration processes.
   * This is a partial specialization for sparse matrices using column storage.
  */
  template<typename Real>
  class ChemicalReactionNetwork<Real, Eigen::SparseMatrix<Real>> {
  public:
    /// Constructor
    ChemicalReactionNetwork(std::vector <MEPBM::ChemicalReaction<Real, Eigen::SparseMatrix<Real>>> reactions,
                            std::vector <MEPBM::ParticleGrowth<Real, Eigen::SparseMatrix<Real>>> growth_processes,
                            std::vector <MEPBM::ParticleAgglomeration<Real, Eigen::SparseMatrix<Real>>> agglomeration_processes)
        : reactions(reactions),
          growth_processes(growth_processes),
          agglomeration_processes(agglomeration_processes) {}

    /// All of the individual reactions in the mechanism
    std::vector <MEPBM::ChemicalReaction<Real, Eigen::SparseMatrix<Real>>> reactions;

    /// All of the growth processes in the mechanism
    std::vector <MEPBM::ParticleGrowth<Real, Eigen::SparseMatrix<Real>>> growth_processes;

    ///All of the agglomeration processes in the mechanism
    std::vector <MEPBM::ParticleAgglomeration<Real, Eigen::SparseMatrix<Real>>> agglomeration_processes;

    /// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, void *)>
    rhs_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void *user_data) {
        int output_value;

        // Make sure x_dot = 0 to begin with
        auto x_dot_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x_dot->content);
        x_dot_vec->setZero();

        // Loop through all individual reactions
        for (auto rxn: reactions) {
          auto rhs = rxn.rhs_function();
          output_value = rhs(time, x, x_dot, user_data);
        }

        // Loop through all growth processes
        for (auto growth: growth_processes) {
          auto rhs = growth.rhs_function();
          output_value = rhs(time, x, x_dot, user_data);
        }

        // Loop through all agglomeration processes
        for (auto agglomeration: agglomeration_processes) {
          auto rhs = agglomeration.rhs_function();
          output_value = rhs(time, x, x_dot, user_data);
        }
        return output_value;
      };
      return fcn;
    }


    /// Returns a function for the Jacobian of the ODE that is compatible with the SUNDIALS API.
    std::function<int(Real, N_Vector, N_Vector, SUNMatrix, void *, N_Vector, N_Vector, N_Vector)>
    jacobian_function() const {
      auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2,
                     N_Vector tmp3) {
        int output_value;

        // Make sure the Jacobian is zero to start -- I'm pretty sure setFromTriplets ignores the previous state but I'm paranoid
        auto J_mat = static_cast<Eigen::SparseMatrix<Real>*>(J->content);
        J_mat->setZero();

        // Create an empty triplet list to gather all non-zero entries in the Jacobian
        std::vector<Eigen::Triplet<Real>> triplet_list;

        // Loop through all individual reactions
        for (auto rxn: reactions) {
          auto jac = rxn.jacobian_function();
          output_value = jac(x, triplet_list, J);
        }

        // Loop through all growth processes
        for (auto growth: growth_processes) {
          auto jac = growth.jacobian_function();
          output_value = jac(x, triplet_list, J);
        }

        // Loop through all agglomeration processes
        for (auto agglomeration: agglomeration_processes) {
          auto jac = agglomeration.jacobian_function();
          output_value = jac(x, triplet_list, J);
        }

        // Fill the Jacobian using the triplet list
        J_mat->setFromTriplets(triplet_list.begin(), triplet_list.end());

        return output_value;
      };
      return fcn;
    }
  };



/**
* This represents a complete system of chemical reactions that form a complete mechanism. The mechanism
* includes a set of individual reactions, a set of growth processes, and a set of agglomeration processes.
 * This is a partial specialization for sparse matrices using row storage.
*/
template<typename Real>
class ChemicalReactionNetwork<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> {
public:
/// Constructor
ChemicalReactionNetwork(std::vector <MEPBM::ChemicalReaction<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>> reactions,
                        std::vector <MEPBM::ParticleGrowth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>> growth_processes,
                        std::vector <MEPBM::ParticleAgglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>> agglomeration_processes)
: reactions(reactions),
  growth_processes(growth_processes),
  agglomeration_processes(agglomeration_processes) {}

/// All of the individual reactions in the mechanism
std::vector <MEPBM::ChemicalReaction<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>> reactions;

/// All of the growth processes in the mechanism
std::vector <MEPBM::ParticleGrowth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>> growth_processes;

///All of the agglomeration processes in the mechanism
std::vector <MEPBM::ParticleAgglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>> agglomeration_processes;

/// Returns a function for the right-hand side of the ODE that is compatible with the SUNDIALS API.
std::function<int(Real, N_Vector, N_Vector, void *)>
rhs_function() const {
  auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, void *user_data) {
    int output_value;

    // Make sure x_dot = 0 to begin with
    auto x_dot_vec = static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(x_dot->content);
    x_dot_vec->setZero();

    // Loop through all individual reactions
    for (auto rxn: reactions) {
      auto rhs = rxn.rhs_function();
      output_value = rhs(time, x, x_dot, user_data);
    }

    // Loop through all growth processes
    for (auto growth: growth_processes) {
      auto rhs = growth.rhs_function();
      output_value = rhs(time, x, x_dot, user_data);
    }

    // Loop through all agglomeration processes
    for (auto agglomeration: agglomeration_processes) {
      auto rhs = agglomeration.rhs_function();
      output_value = rhs(time, x, x_dot, user_data);
    }
    return output_value;
  };
  return fcn;
}


/// Returns a function for the Jacobian of the ODE that is compatible with the SUNDIALS API.
std::function<int(Real, N_Vector, N_Vector, SUNMatrix, void *, N_Vector, N_Vector, N_Vector)>
jacobian_function() const {
  auto fcn = [&](Real time, N_Vector x, N_Vector x_dot, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2,
                 N_Vector tmp3) {
    int output_value;

    // Make sure the Jacobian is zero to start -- I'm pretty sure setFromTriplets ignores the previous state but I'm paranoid
    auto J_mat = static_cast<Eigen::SparseMatrix<Real, Eigen::RowMajor>*>(J->content);
    J_mat->setZero();

    // Create an empty triplet list to gather all non-zero entries in the Jacobian
    std::vector<Eigen::Triplet<Real>> triplet_list;

    // Loop through all individual reactions
    for (auto rxn: reactions) {
      auto jac = rxn.jacobian_function();
      output_value = jac(x, triplet_list, J);
    }

    // Loop through all growth processes
    for (auto growth: growth_processes) {
      auto jac = growth.jacobian_function();
      output_value = jac(x, triplet_list, J);
    }

    // Loop through all agglomeration processes
    for (auto agglomeration: agglomeration_processes) {
      auto jac = agglomeration.jacobian_function();
      output_value = jac(x, triplet_list, J);
    }

    // Fill the Jacobian using the triplet list
    J_mat->setFromTriplets(triplet_list.begin(), triplet_list.end());

    return output_value;
  };
  return fcn;
}
};
}

#endif //MEPBM_CHEMICAL_REACTION_NETWORK_H
