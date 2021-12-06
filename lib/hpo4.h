#ifndef MEPBM_HPO4_H
#define MEPBM_HPO4_H

#include <vector>
#include <cmath>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iostream>
#include "models.h"


namespace Model
{
  /**
   * Nucleation 1
   *    A + 2solv <-> Asolv + L
   *    Asolv -> B(2) + L
   *
   *    A is size 2
   *    Asolv is size 2
   */
  template<typename Real, typename Matrix>
  class HPO4NucleationMechanismA2Solv : public Model::RightHandSideContribution<Real, Matrix>
  {
  public:
    const unsigned int A_index, As_index, ligand_index, particle_index;
    const Real rate_forward, rate_backward, rate_nucleation;
    const Real solvent;

    /// Constructor -- creates an invalid object by default
    TermolecularNucleation();

    /// Constructor
    TermolecularNucleation(unsigned int A_index, unsigned int As_index, unsigned int ligand_index,
                           unsigned int particle_index,
                           Real rate_forward, Real rate_backward, Real rate_nucleation,
                           Real solvent);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Matrix &J) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;
  };



  /**
   * Nucleation 2
   *    A + 4solv <-> 2Asolv + 2L
   *    Asolv -> B(1)
   *
   *    A is size 2
   *    Asolv is size 1
   */



  /**
   * Mechanism 1A -- Nucleation 1 + growth instigated by A
   *    A + 2solv <-> Asolv + L
   *    Asolv -> B(2) + L
   *    A + B -> B(+2) + 2L
   *    A + C -> C(+2) + 2L
   *    B(i) + B(j) -> B(i+j)
   */



  /**
   * Mechanism 1B -- Nucleation 1 + growth instigated by Asolv
   *    A + 2solv <-> Asolv + L
   *    Asolv -> B(2) + L
   *    Asolv + B -> B(+2) + 2L
   *    Asolv + C -> C(+2) + 2L
   *    B(i) + B(j) -> B(i+j)
   */



  /**
   * Mechanism 2A -- Nucleation 2 + growth instigated by A
   *    A + 4solv <-> 2Asolv + 2L
   *    Asolv -> B(1)
   *    A + B -> B(+2) + 2L
   *    A + C -> C(+2) + 2L
   *    B(i) + B(j) -> B(i+j)
   */



  /**
   * Mechanism 2B -- Nucleation 2 + growth instigated by Asolv
   *    A + 4solv <-> 2Asolv + 2L
   *    Asolv -> B(1)
   *    Asolv + B -> B(+2) *Note no ligand created*
   *    Asolv + C -> C(+2) *Note no ligand created*
   *    B(i) + B(j) -> B(i+j)
   */
}




#endif //MEPBM_HPO4_H
