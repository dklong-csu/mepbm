#ifndef MEPBM_MODELS_H
#define MEPBM_MODELS_H

/*
 * DEPRECATED -- only kept to maintain reproducibility of published results
 */

/*
 * Mechanism-Enabled Population Balance Model (MEPBM)
 *
 * This is a class which models the behavior of the nucleation and growth process for nanoparticle systems.
 * The model is a system of ordinary differential equations (ODEs), which will be solved numerically.
 * This class essentially is a way to create the right hand side and Jacobian of the ODE. This class is setup
 * so that it can easily be connected to Boost ODE solvers.
 *
 * In MEPBM theory there are many mechanisms which correspond to different ODE models. To avoid having to write
 * a unique class for each mechanism, the class instead has a modular formation based on high level features
 * that the mechanisms are composed of. A mechanism can be identified by a nucleation mechanism and a series of
 * growth and agglomeration steps which occur at specified particle ranges.
 *
 * The class is created such that the nucleation mechanism, growth, and agglomeration can be added into the model
 * as contributions to the right hand side of the ODE. Therefore, any mechanism can be constructed using a nucleation
 * mechanism and a series of growth and agglomeration steps.
 *
 * Nucleation mechanism -- this is the most variable since it will be unique for each chemical system. As such, the
 * creation of this is likely best included in your program rather than in the library.
 *
 * Growth -- this can be uniquely identified by providing a particle size range (smallest and largest size) for which
 * the growth applies to, the rate the growth occurs, and the variable corresponding to the precursor.
 *
 * Agglomeration -- this can be uniquely identified by providing a particle size range (smallest and largest size)
 * for which the growth applies to, and the rate the growth occurs.
 */

#include <vector>
#include <cmath>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iostream>
#include <cassert>

namespace Model
{
/*
 * A function which maps the size of a particle -- i.e. the number of molecules in the particle -- and
 * returns the number of ``available atoms." The concept of ``available atoms" is a way of describing
 * how each molecule on the outside of the particle represents a binding site for which a precursor
 * (in the growth phase) or another particle (in the agglomeration phase) can attach to. This acts to
 * hasten the speed of the reaction because if there are 5 binding sites, then it is reasonable that
 * the reaction would occur 5 times as quickly as if there were only 1 binding site.
 *
 * We make the assumption that a particle is spherical. Based on work by Schmidt and Smirnov,
 * we use the relationship
 *      r(size) = 2.677 * size^(-0.28)
 * to describe the proportion of molecules in a particle whose surface is on the boundary of the particle.
 * To get the ``available atoms" we then say (number of molecules)*r(size).
 * If the molecule being tracked in the particle is a monomer (one atom) then: size = number of molecules.
 * If the molecule being tracked is a dimer (two atoms) then: size = 2 * number of molecules.
 * If the molecule being tracked is m-atoms large then: size = m * number of molecules.
 */
  ///
  /// A function that converts the diameter of a particle as measured using TEM to the number of atoms present in the particle
  ///
  template<typename Real>
  Real atoms(unsigned int size);


  template<typename Real>
  Real atoms(unsigned int size)
  {
    return 2.677 * size * std::pow(1.*size, -0.28);
  }



  ///
  /// A base class for describing the effects of nucleation, growth, and agglomeration
  ///
  template<typename Real, typename Matrix>
  class RightHandSideContribution
  {
  public:
    /// Function for adding the effects of this object to the right-hand side vector of the system of ODEs
    virtual void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                         Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) = 0;

    /// Function for adding the effects of this object ot the Jacobian of the system of ODEs
    virtual void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                              Matrix &J) = 0;

    /// Helper function for determining the sparsity pattern when using sparse matrices for the Jacobian
    virtual void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) = 0;

    /// Helper function for estimating how many nonzero elements are in the Jacobian when using sparse matrices
    virtual void update_num_nonzero(unsigned int &num_nonzero) = 0;

  };




  ///
  /// An object describing the effect of termolecular nucleation as described by the reactions
  /// A + 2S <-> A_s + L
  /// 2A_s + A -> B + L
  ///
  template<typename Real, typename Matrix>
  class TermolecularNucleation : public Model::RightHandSideContribution<Real, Matrix>
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



  /// Partial specialization for a dense matrix
  template<typename Real>
  class TermolecularNucleation<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
      : public Model::RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
  {
  public:
    const unsigned int A_index, As_index, ligand_index, particle_index;
    const Real rate_forward, rate_backward, rate_nucleation;
    const Real solvent;


    TermolecularNucleation();

    TermolecularNucleation(unsigned int A_index, unsigned int As_index, unsigned int ligand_index,
                           unsigned int particle_index,
                           Real rate_forward, Real rate_backward, Real rate_nucleation,
                           Real solvent);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &J) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) {}

    void update_num_nonzero(unsigned int &num_nonzero) {}
  };


  template<typename Real, typename Matrix>
  TermolecularNucleation<Real, Matrix>::TermolecularNucleation()
      : TermolecularNucleation(std::numeric_limits<unsigned int>::signaling_NaN(),
                               std::numeric_limits<unsigned int>::signaling_NaN(),
                               std::numeric_limits<unsigned int>::signaling_NaN(),
                               std::numeric_limits<unsigned int>::signaling_NaN(),
                               std::numeric_limits<Real>::signaling_NaN(),
                               std::numeric_limits<Real>::signaling_NaN(),
                               std::numeric_limits<Real>::signaling_NaN(),
                               std::numeric_limits<Real>::signaling_NaN())
  {}



  template<typename Real>
  TermolecularNucleation<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::TermolecularNucleation(
      const unsigned int A_index,
      const unsigned int As_index,
      const unsigned int ligand_index,
      const unsigned int particle_index,
      const Real rate_forward,
      const Real rate_backward,
      const Real rate_nucleation,
      const Real solvent)
      : A_index(A_index), As_index(As_index), ligand_index(ligand_index), particle_index(particle_index)
      , rate_forward(rate_forward), rate_backward(rate_backward), rate_nucleation(rate_nucleation)
      , solvent(solvent)
  {}



  template<typename Real>
  void TermolecularNucleation<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    const Real diss_forward = rate_forward * x(A_index) * solvent*solvent;
    const Real diss_backward = rate_backward * x(As_index) * x(ligand_index);
    const Real nucleation = rate_nucleation * x(A_index) * x(As_index) * x(As_index);

    rhs(A_index) += -diss_forward + diss_backward - nucleation;
    rhs(As_index) += diss_forward - diss_backward - 2 * nucleation;
    rhs(ligand_index) += diss_forward - diss_backward + nucleation;
    rhs(particle_index) += nucleation;
  }



  template<typename Real>
  void
  TermolecularNucleation<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_contribution_to_jacobian(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &jacobi)
  {
    const Real diss_forward_dA = rate_forward * solvent * solvent;

    const Real diss_backward_dAs = rate_backward * x(ligand_index);
    const Real diss_backward_dL = rate_backward * x(As_index);

    const Real nucleation_dA = rate_nucleation * x(As_index) * x(As_index);
    const Real nucleation_dAs = 2 * rate_nucleation * x(A_index) * x(As_index);

    jacobi(A_index, A_index) += -diss_forward_dA - nucleation_dA;
    jacobi(A_index, As_index) += diss_backward_dAs - nucleation_dAs;
    jacobi(A_index, ligand_index) += diss_backward_dL;

    jacobi(As_index, A_index) += diss_forward_dA - 2 * nucleation_dA;
    jacobi(As_index, As_index) += -diss_backward_dAs - 2 * nucleation_dAs;
    jacobi(As_index, ligand_index) += -diss_backward_dL;

    jacobi(ligand_index, A_index) += diss_forward_dA + nucleation_dA;
    jacobi(ligand_index, As_index) += -diss_backward_dAs + nucleation_dAs;
    jacobi(ligand_index, ligand_index) += -diss_backward_dL;

    jacobi(particle_index, A_index) += nucleation_dA;
    jacobi(particle_index, As_index) += nucleation_dAs;
  }



  /// Partial specialization for a sparse matrix
  template<typename Real>
  class TermolecularNucleation<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
      : public Model::RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
  {
  public:
    const unsigned int A_index, As_index, ligand_index, particle_index;
    const Real rate_forward, rate_backward, rate_nucleation;
    const Real solvent;

    TermolecularNucleation();

    TermolecularNucleation(unsigned int A_index, unsigned int As_index, unsigned int ligand_index,
                           unsigned int particle_index,
                           Real rate_forward, Real rate_backward, Real rate_nucleation,
                           Real solvent);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::SparseMatrix<Real, Eigen::RowMajor> &J) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;
  };



  template<typename Real>
  TermolecularNucleation<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::TermolecularNucleation(
      const unsigned int A_index,
      const unsigned int As_index,
      const unsigned int ligand_index,
      const unsigned int particle_index,
      const Real rate_forward,
      const Real rate_backward,
      const Real rate_nucleation,
      const Real solvent)
      : A_index(A_index), As_index(As_index), ligand_index(ligand_index), particle_index(particle_index)
      , rate_forward(rate_forward), rate_backward(rate_backward), rate_nucleation(rate_nucleation)
      , solvent(solvent)
  {}



  template<typename Real>
  void TermolecularNucleation<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    const Real diss_forward = rate_forward * x(A_index) * solvent*solvent;
    const Real diss_backward = rate_backward * x(As_index) * x(ligand_index);
    const Real nucleation = rate_nucleation * x(A_index) * x(As_index) * x(As_index);

    rhs(A_index) += -diss_forward + diss_backward - nucleation;
    rhs(As_index) += diss_forward - diss_backward - 2 * nucleation;
    rhs(ligand_index) += diss_forward - diss_backward + nucleation;
    rhs(particle_index) += nucleation;
  }


  template<typename Real>
  void
  TermolecularNucleation<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_contribution_to_jacobian(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::SparseMatrix<Real, Eigen::RowMajor> &jacobi)
  {
    const Real diss_forward_dA = rate_forward * solvent * solvent;

    const Real diss_backward_dAs = rate_backward * x(ligand_index);
    const Real diss_backward_dL = rate_backward * x(As_index);

    const Real nucleation_dA = rate_nucleation * x(As_index) * x(As_index);
    const Real nucleation_dAs = 2 * rate_nucleation * x(A_index) * x(As_index);

    jacobi.coeffRef(A_index, A_index) += -diss_forward_dA - nucleation_dA;
    jacobi.coeffRef(A_index, As_index) += diss_backward_dAs - nucleation_dAs;
    jacobi.coeffRef(A_index, ligand_index) += diss_backward_dL;

    jacobi.coeffRef(As_index, A_index) += diss_forward_dA - 2 * nucleation_dA;
    jacobi.coeffRef(As_index, As_index) += -diss_backward_dAs - 2 * nucleation_dAs;
    jacobi.coeffRef(As_index, ligand_index) += -diss_backward_dL;

    jacobi.coeffRef(ligand_index, A_index) += diss_forward_dA + nucleation_dA;
    jacobi.coeffRef(ligand_index, As_index) += -diss_backward_dAs + nucleation_dAs;
    jacobi.coeffRef(ligand_index, ligand_index) += -diss_backward_dL;

    jacobi.coeffRef(particle_index, A_index) += nucleation_dA;
    jacobi.coeffRef(particle_index, As_index) += nucleation_dAs;
  }


  template<typename Real>
  void
  TermolecularNucleation<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list)
  {
    triplet_list.push_back(Eigen::Triplet<Real>(A_index, A_index));
    triplet_list.push_back(Eigen::Triplet<Real>(A_index, As_index));
    triplet_list.push_back(Eigen::Triplet<Real>(A_index, ligand_index));

    triplet_list.push_back(Eigen::Triplet<Real>(As_index, A_index));
    triplet_list.push_back(Eigen::Triplet<Real>(As_index, As_index));
    triplet_list.push_back(Eigen::Triplet<Real>(As_index, ligand_index));

    triplet_list.push_back(Eigen::Triplet<Real>(ligand_index, A_index));
    triplet_list.push_back(Eigen::Triplet<Real>(ligand_index, As_index));
    triplet_list.push_back(Eigen::Triplet<Real>(ligand_index, ligand_index));

    triplet_list.push_back(Eigen::Triplet<Real>(particle_index, A_index));
    triplet_list.push_back(Eigen::Triplet<Real>(particle_index, As_index));
  }


  template<typename Real>
  void
  TermolecularNucleation<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::update_num_nonzero(unsigned int &num_nonzero)
  {
    num_nonzero += 11;
  }



  ///
  /// An object describing the growth of particles (B) via the reaction
  /// A + B -> C + L
  ///
  template<typename Real, typename Matrix>
  class Growth : public RightHandSideContribution<Real, Matrix>
  {
  public:
    const unsigned int A_index, smallest_size, largest_size, max_size, ligand_index, conserved_size;
    const Real rate;
    const unsigned int smallest_size_index;

    Growth();

    Growth(unsigned int A_index, unsigned int smallest_size, unsigned int largest_size,
           unsigned int max_size, unsigned int ligand_index, unsigned int conserved_size,
           Real rate, unsigned int smallest_size_index);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Matrix &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;

  private:
    unsigned int index_to_size(unsigned int size);

    unsigned int size_to_index(unsigned int index);
  };



  /// Partial specialization for a dense matrix
  template<typename Real>
  class Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
      : public RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
  {
  public:
    const unsigned int A_index, smallest_size, largest_size, max_size, ligand_index, conserved_size;
    const Real rate;
    const unsigned int smallest_size_index;

    Growth();

    Growth(unsigned int A_index, unsigned int smallest_size, unsigned int largest_size,
           unsigned int max_size, unsigned int ligand_index, unsigned int conserved_size,
           Real rate, unsigned int smallest_size_index);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) {}

    void update_num_nonzero(unsigned int &num_nonzero) {}

  private:
    unsigned int index_to_size(unsigned int size);

    unsigned int size_to_index(unsigned int index);
  };



  template<typename Real>
  unsigned int
  Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::index_to_size(unsigned int index)
  {
    return (index - smallest_size_index) + smallest_size;
  }



  template<typename Real>
  unsigned int
  Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::size_to_index(unsigned int size)
  {
    return (size - smallest_size) + smallest_size_index;
  }



  template<typename Real>
  Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Growth()
      : Growth(std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<Real>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN())
  {}



  template<typename Real>
  Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Growth(
      const unsigned int A_index,
      const unsigned int smallest_size,
      const unsigned int largest_size,
      const unsigned int max_size,
      const unsigned int ligand_index,
      const unsigned int conserved_size,
      const Real rate,
      const unsigned int smallest_index)
      : A_index(A_index)
      , smallest_size(smallest_size)
      , largest_size(largest_size)
      , max_size(max_size)
      , ligand_index(ligand_index)
      , conserved_size(conserved_size)
      , rate(rate)
      , smallest_size_index(smallest_index)
  {}



  template<typename Real>
  void Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    assert(smallest_size < largest_size);
    assert(largest_size <= max_size);
    auto first_index = size_to_index(smallest_size);
    auto last_index = size_to_index(largest_size);
    assert(first_index <= last_index);
    assert(first_index != A_index);
    assert(first_index != ligand_index);
    assert(last_index != A_index);
    assert(last_index != ligand_index);
    assert(!(first_index < A_index && A_index < last_index));
    assert(!(first_index < ligand_index && ligand_index < last_index));
    for (unsigned int index = first_index; index <= last_index; ++index)
    {
      const Real rxn_factor = rate *  x(A_index) * atoms<Real>(index_to_size(index)) * x(index_to_size(index));
      rhs(index) -= rxn_factor;
      rhs(ligand_index) += rxn_factor;
      rhs(A_index) -= rxn_factor;

      auto created_size = index_to_size(index) + conserved_size;
      if (created_size <= max_size)
      {
        auto created_index = size_to_index(created_size);
        rhs(created_index) += rxn_factor;
      }
    }
  }



  template<typename Real>
  void
  Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_contribution_to_jacobian(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &jacobi)
  {
    assert(smallest_size < largest_size);
    assert(largest_size <= max_size);
    auto first_index = size_to_index(smallest_size);
    auto last_index = size_to_index(largest_size);
    assert(first_index <= last_index);
    assert(first_index != A_index);
    assert(first_index != ligand_index);
    assert(last_index != A_index);
    assert(last_index != ligand_index);
    assert(!(first_index < A_index && A_index < last_index));
    assert(!(first_index < ligand_index && ligand_index < last_index));

    for (unsigned int index = first_index; index <= last_index; ++index)
    {
      const Real rxn_factor_dA = rate * atoms<Real>(index_to_size(index)) * x(index);
      const Real rxn_factor_dn = rate * x(A_index) * atoms<Real>(index_to_size(index));

      jacobi(index, A_index) -= rxn_factor_dA;
      jacobi(index, index) -= rxn_factor_dn;

      jacobi(ligand_index, A_index) += rxn_factor_dA;
      jacobi(ligand_index, index) += rxn_factor_dn;

      jacobi(A_index, A_index) -= rxn_factor_dA;
      jacobi(A_index, index) -= rxn_factor_dn;

      auto created_size = index_to_size(index) + conserved_size;
      if (created_size <= max_size)
      {
        auto created_index = size_to_index(created_size);
        jacobi(created_index, A_index) += rxn_factor_dA;
        jacobi(created_index, index) += rxn_factor_dn;
      }
    }
  }



  /// Partial specialization for a sparse matrix
  template<typename Real>
  class Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
      : public RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
  {
  public:
    const unsigned int A_index, smallest_size, largest_size, max_size, ligand_index, conserved_size;
    const Real rate;
    const unsigned int smallest_size_index;

    Growth();

    Growth(unsigned int A_index, unsigned int smallest_size, unsigned int largest_size,
           unsigned int max_size, unsigned int ligand_index, unsigned int conserved_size,
           Real rate, unsigned int smallest_size_index);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::SparseMatrix<Real, Eigen::RowMajor> &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;

  private:
    unsigned int index_to_size(unsigned int size);

    unsigned int size_to_index(unsigned int index);
  };



  template<typename Real>
  unsigned int
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::index_to_size(unsigned int index)
  {
    return (index - smallest_size_index) + smallest_size;
  }



  template<typename Real>
  unsigned int
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::size_to_index(unsigned int size)
  {
    return (size - smallest_size) + smallest_size_index;
  }



  template<typename Real>
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Growth()
      : Growth(std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<Real>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN())
  {}



  template<typename Real>
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Growth(const unsigned int A_index,
                                                  const unsigned int smallest_size,
                                                  const unsigned int largest_size,
                                                  const unsigned int max_size,
                                                  const unsigned int ligand_index,
                                                  const unsigned int conserved_size,
                                                  const Real rate,
                                                  const unsigned int smallest_index)
      : A_index(A_index)
      , smallest_size(smallest_size)
      , largest_size(largest_size)
      , max_size(max_size)
      , ligand_index(ligand_index)
      , conserved_size(conserved_size)
      , rate(rate)
      , smallest_size_index(smallest_index)
  {}



  template<typename Real>
  void Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    assert(smallest_size < largest_size);
    assert(largest_size <= max_size);
    auto first_index = size_to_index(smallest_size);
    auto last_index = size_to_index(largest_size);
    assert(first_index <= last_index);
    assert(first_index != A_index);
    assert(first_index != ligand_index);
    assert(last_index != A_index);
    assert(last_index != ligand_index);
    assert(!(first_index < A_index && A_index < last_index));
    assert(!(first_index < ligand_index && ligand_index < last_index));
    for (unsigned int index = first_index; index <= last_index; ++index)
    {
      const Real rxn_factor = rate *  x(A_index) * atoms<Real>(index_to_size(index)) * x(index);
      rhs(index) -= rxn_factor;
      rhs(ligand_index) += rxn_factor;
      rhs(A_index) -= rxn_factor;

      auto created_size = index_to_size(index) + conserved_size;
      if (created_size <= max_size)
      {
        auto created_index = size_to_index(created_size);
        rhs(created_index) += rxn_factor;
      }
    }
  }



  template<typename Real>
  void
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_contribution_to_jacobian(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::SparseMatrix<Real, Eigen::RowMajor> &jacobi)
  {
    assert(smallest_size < largest_size);
    assert(largest_size <= max_size);
    auto first_index = size_to_index(smallest_size);
    auto last_index = size_to_index(largest_size);
    assert(first_index <= last_index);
    assert(first_index != A_index);
    assert(first_index != ligand_index);
    assert(last_index != A_index);
    assert(last_index != ligand_index);
    assert(!(first_index < A_index && A_index < last_index));
    assert(!(first_index < ligand_index && ligand_index < last_index));

    for (unsigned int index = first_index; index <= last_index; ++index)
    {
      const Real rxn_factor_dA = rate * atoms<Real>(index_to_size(index)) * x(index);
      const Real rxn_factor_dn = rate * x(A_index) * atoms<Real>(index_to_size(index));

      jacobi.coeffRef(index, A_index) -= rxn_factor_dA;
      jacobi.coeffRef(index, index) -= rxn_factor_dn;

      jacobi.coeffRef(ligand_index, A_index) += rxn_factor_dA;
      jacobi.coeffRef(ligand_index, index) += rxn_factor_dn;

      jacobi.coeffRef(A_index, A_index) -= rxn_factor_dA;
      jacobi.coeffRef(A_index, index) -= rxn_factor_dn;

      auto created_size = index_to_size(index) + conserved_size;
      if (created_size <= max_size)
      {
        auto created_index = size_to_index(created_size);
        jacobi.coeffRef(created_index, A_index) += rxn_factor_dA;
        jacobi.coeffRef(created_index, index) += rxn_factor_dn;
      }
    }
  }



  template<typename Real>
  void
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list)
  {
    assert(smallest_size < largest_size);
    assert(largest_size <= max_size);
    auto first_index = size_to_index(smallest_size);
    auto last_index = size_to_index(largest_size);
    assert(first_index <= last_index);
    assert(first_index != A_index);
    assert(first_index != ligand_index);
    assert(last_index != A_index);
    assert(last_index != ligand_index);
    assert(!(first_index < A_index && A_index < last_index));
    assert(!(first_index < ligand_index && ligand_index < last_index));

    for (unsigned int index = first_index; index <= last_index; ++index)
    {
      triplet_list.push_back(Eigen::Triplet<Real>(index, A_index));
      triplet_list.push_back(Eigen::Triplet<Real>(index, index));

      triplet_list.push_back(Eigen::Triplet<Real>(ligand_index, A_index));
      triplet_list.push_back(Eigen::Triplet<Real>(ligand_index, index));

      triplet_list.push_back(Eigen::Triplet<Real>(A_index, A_index));
      triplet_list.push_back(Eigen::Triplet<Real>(A_index, index));

      auto created_size = index_to_size(index) + conserved_size;
      if (created_size <= max_size)
      {
        auto created_index = size_to_index(created_size);
        triplet_list.push_back(Eigen::Triplet<Real>(created_index, A_index));
        triplet_list.push_back(Eigen::Triplet<Real>(created_index, index));
      }
    }
  }



  template<typename Real>
  void
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::update_num_nonzero(unsigned int &num_nonzero)
  {
    num_nonzero += (largest_size - smallest_size) * 8;
  }



  ///
  /// An object describing particle agglomeration via the reaction
  /// B + C -> D
  ///
  template<typename Real, typename Matrix>
  class Agglomeration : public RightHandSideContribution<Real, Matrix>
  {
  public:
    const unsigned int B_smallest_size, B_largest_size;
    const unsigned int C_smallest_size, C_largest_size;
    const unsigned int max_size;
    const unsigned int conserved_size;
    const Real rate;
    const unsigned int first_B_index;

    Agglomeration();

    Agglomeration(unsigned int B_smallest_size, unsigned int B_largest_size,
                  unsigned int C_smallest_size, unsigned int C_largest_size,
                  unsigned int max_size, unsigned int conserved_size, Real rate,
                  unsigned int B_smallest_index);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Matrix &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;

  private:
    unsigned int index_to_size(unsigned int size);

    unsigned int size_to_index(unsigned int index);
  };



  /// Partial specialization for a dense matrix
  template<typename Real>
  class Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
      : public RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
  {
  public:
    const unsigned int B_smallest_size, B_largest_size;
    const unsigned int C_smallest_size, C_largest_size;
    const unsigned int max_size;
    const unsigned int conserved_size;
    const Real rate;
    const unsigned int first_B_index;

    Agglomeration();

    Agglomeration(unsigned int B_smallest_size, unsigned int B_largest_size,
                  unsigned int C_smallest_size, unsigned int C_largest_size,
                  unsigned int max_size, unsigned int conserved_size, Real rate,
                  unsigned int smallest_particle_index);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) {}

    void update_num_nonzero(unsigned int &num_nonzero) {}

  private:
    unsigned int index_to_size(unsigned int size);

    unsigned int size_to_index(unsigned int index);
  };



  template<typename Real>
  unsigned int
  Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::index_to_size(unsigned int index)
  {
    return (index - first_B_index) + B_smallest_size;
  }



  template<typename Real>
  unsigned int
  Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::size_to_index(unsigned int size)
  {
    return (size - B_smallest_size) + first_B_index;
  }



  template<typename Real>
  Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Agglomeration()
      : Agglomeration(std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<Real>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN())
  {}



  template<typename Real>
  Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Agglomeration(
      const unsigned int B_smallest_size,
      const unsigned int B_largest_size,
      const unsigned int C_smallest_size,
      const unsigned int C_largest_size,
      const unsigned int max_size,
      const unsigned int conserved_size,
      const Real rate,
      const unsigned int smallest_B_index)
      : B_smallest_size(B_smallest_size), B_largest_size(B_largest_size)
      , C_smallest_size(C_smallest_size), C_largest_size(C_largest_size)
      , max_size(max_size)
      , conserved_size(conserved_size)
      , rate(rate)
      , first_B_index(smallest_B_index)
  {}



  template<typename Real>
  void Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    assert(x.size() == rhs.size());
    assert(B_smallest_size < B_largest_size);
    assert(C_smallest_size < C_largest_size);
    assert(B_largest_size <= max_size);
    assert(C_largest_size <= max_size);
    auto index0_B = size_to_index(B_smallest_size);
    auto index1_B = size_to_index(B_largest_size);
    auto index0_C = size_to_index(C_smallest_size);
    auto index1_C = size_to_index(C_largest_size);
    assert(index0_B < index1_B);
    assert(index0_C < index1_C);
    assert(index1_B < x.size());
    assert(index1_C < x.size());

    // Pre-calculate terms of the derivative that will be used since we require an expensive double loop.
    // Many of the terms will be calculated more than once if they are not pre-calculated, which adds
    // up as the double loop gets larger.
    //
    // The factor is calculated to be the number of binding sites * current concentration
    std::vector<Real> rxn_factors(x.size(), 0.);
    for (unsigned int i=index0_B; i<=index1_B; ++i)
    {
      rxn_factors[i] = atoms<Real>(index_to_size(i)) * x(i);
    }

    if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
    {
      for (unsigned int i=index0_C; i<=index1_C; ++i)
      {
        rxn_factors[i] = atoms<Real>(index_to_size(i)) * x(i);
      }
    }

    // Calculate the right-hand side contributions from agglomeration
    // Two particles interact with each other, one B particle and one C particle.
    // The rate this occurs based on how frequently a B particle and C particle will interact
    // with each other as well as by how quickly the reaction occurs.
    // The frequency of the particles meeting is proportional to the product of the rxn_factors calculated above.
    // Then the provided rate constant, rate, scales the product to appropriately compute the reaction rate.
    //
    // We always track the outflow due to agglomeration. However, agglomeration might yield a particle whose
    // size is larger than we track. For example, if the max particle size tracked is 10 and agglomeration
    // occurs between a particle of size 6 and 7, then the loss of size 6 and 7 particles will be calculated
    // but the gain of size 13 particles will not because only particles up to size 10 are tracked.
    for (unsigned int i = index0_B; i <= index1_B; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(index0_C, i); j <= index1_C; ++j)
      {
        const auto rxn_deriv = rate * rxn_factors[i] * rxn_factors[j];
        rhs(i) -= rxn_deriv;
        rhs(j) -= rxn_deriv;

        auto created_size = size_to_index(i) + size_to_index(j);
        if (created_size <= max_size)
        {
          auto created_index = size_to_index(created_size);
          rhs(created_index) += rxn_deriv;
        }
      }
    }
  }



  template<typename Real>
  void
  Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_contribution_to_jacobian(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real,Eigen::Dynamic, Eigen::Dynamic> &jacobi)
  {
    assert(B_smallest_size < B_largest_size);
    assert(C_smallest_size < C_largest_size);
    assert(B_largest_size <= max_size);
    assert(C_largest_size <= max_size);
    auto index0_B = size_to_index(B_smallest_size);
    auto index1_B = size_to_index(B_largest_size);
    auto index0_C = size_to_index(C_smallest_size);
    auto index1_C = size_to_index(C_largest_size);
    assert(index0_B < index1_B);
    assert(index0_C < index1_C);
    assert(index1_B < x.size());
    assert(index1_C < x.size());

    std::vector<Real> rxn_factors(x.size(), 0.);
    std::vector<Real> rxn_factors_dn(x.size(), 0.);
    for (unsigned int i=index0_B; i<=index1_B; ++i)
    {
      rxn_factors[i] = atoms<Real>(index_to_size(i)) * x(i);
      rxn_factors_dn[i] = atoms<Real>(index_to_size(i));
    }

    if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
    {
      for (unsigned int i=index0_C; i<=index1_C; ++i)
      {
        rxn_factors[i] = atoms<Real>(index_to_size(i)) * x(i);
        rxn_factors_dn[i] = atoms<Real>(index_to_size(i));
      }
    }

    for (unsigned int i = index0_B; i <= index1_B; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(index0_C, i); j <= index1_C; ++j)
      {
        const auto rxn_deriv_i = rate * rxn_factors_dn[i] * rxn_factors[j];
        const auto rxn_deriv_j = rate * rxn_factors[i] * rxn_factors_dn[j];

        jacobi(i, i) -= rxn_deriv_i;
        jacobi(i, j) -= rxn_deriv_j;

        jacobi(j, i) -= rxn_deriv_i;
        jacobi(j, j) -= rxn_deriv_j;

        auto created_size = index_to_size(i) + index_to_size(j);
        if (created_size <= max_size)
        {
          auto created_index = size_to_index(created_size);
          jacobi(created_index, i) += rxn_deriv_i;
          jacobi(created_index, j) += rxn_deriv_j;
        }
      }
    }
  }



  /// Partial specialization for a sparse matrix
  template<typename Real>
  class Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
      : public RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
  {
  public:
    const unsigned int B_smallest_size, B_largest_size;
    const unsigned int C_smallest_size, C_largest_size;
    const unsigned int max_size;
    const unsigned int conserved_size;
    const Real rate;
    const unsigned int first_B_index;

    Agglomeration();

    Agglomeration(unsigned int B_smallest_size, unsigned int B_largest_size,
                  unsigned int C_smallest_size, unsigned int C_largest_size,
                  unsigned int max_size, unsigned int conserved_size, Real rate,
                  unsigned int smallest_particle_index);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::SparseMatrix<Real, Eigen::RowMajor> &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;

  private:
    unsigned int index_to_size(unsigned int size);

    unsigned int size_to_index(unsigned int index);
  };



  template<typename Real>
  unsigned int
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::index_to_size(unsigned int index)
  {
    return (index - first_B_index) + B_smallest_size;
  }



  template<typename Real>
  unsigned int
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::size_to_index(unsigned int size)
  {
    return (size - B_smallest_size) + first_B_index;
  }



  template<typename Real>
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Agglomeration()
      : Agglomeration(std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<Real>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN())
  {}



  template<typename Real>
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Agglomeration(
      const unsigned int B_smallest_size,
      const unsigned int B_largest_size,
      const unsigned int C_smallest_size,
      const unsigned int C_largest_size,
      const unsigned int max_size,
      const unsigned int conserved_size,
      const Real rate,
      const unsigned int smallest_B_index)
      : B_smallest_size(B_smallest_size), B_largest_size(B_largest_size)
      , C_smallest_size(C_smallest_size), C_largest_size(C_largest_size)
      , max_size(max_size)
      , conserved_size(conserved_size)
      , rate(rate)
      , first_B_index(smallest_B_index)
  {}



  template<typename Real>
  void Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    assert(x.size() == rhs.size());
    assert(B_smallest_size < B_largest_size);
    assert(C_smallest_size < C_largest_size);
    assert(B_largest_size <= max_size);
    assert(C_largest_size <= max_size);
    auto index0_B = size_to_index(B_smallest_size);
    auto index1_B = size_to_index(B_largest_size);
    auto index0_C = size_to_index(C_smallest_size);
    auto index1_C = size_to_index(C_largest_size);
    assert(index0_B < index1_B);
    assert(index0_C < index1_C);
    assert(index1_B < x.size());
    assert(index1_C < x.size());

    // Pre-calculate terms of the derivative that will be used since we require an expensive double loop.
    // Many of the terms will be calculated more than once if they are not pre-calculated, which adds
    // up as the double loop gets larger.
    //
    // The factor is calculated to be the number of binding sites * current concentration
    std::vector<Real> rxn_factors(x.size(), 0.);
    for (unsigned int i=index0_B; i<=index1_B; ++i)
    {
      rxn_factors[i] = atoms<Real>(index_to_size(i)) * x(i);
    }

    if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
    {
      for (unsigned int i=index0_C; i<=index1_C; ++i)
      {
        rxn_factors[i] = atoms<Real>(index_to_size(i)) * x(i);
      }
    }

    // Calculate the right-hand side contributions from agglomeration
    // Two particles interact with each other, one B particle and one C particle.
    // The rate this occurs based on how frequently a B particle and C particle will interact
    // with each other as well as by how quickly the reaction occurs.
    // The frequency of the particles meeting is proportional to the product of the rxn_factors calculated above.
    // Then the provided rate constant, rate, scales the product to appropriately compute the reaction rate.
    //
    // We always track the outflow due to agglomeration. However, agglomeration might yield a particle whose
    // size is larger than we track. For example, if the max particle size tracked is 10 and agglomeration
    // occurs between a particle of size 6 and 7, then the loss of size 6 and 7 particles will be calculated
    // but the gain of size 13 particles will not because only particles up to size 10 are tracked.
    for (unsigned int i = index0_B; i <= index1_B; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(index0_C, i); j <= index1_C; ++j)
      {
        const auto rxn_deriv = rate * rxn_factors[i] * rxn_factors[j];
        rhs(i) -= rxn_deriv;
        rhs(j) -= rxn_deriv;

        auto created_size = index_to_size(i) + index_to_size(j);
        if (created_size <= max_size)
        {
          auto created_index = size_to_index(created_size);
          rhs(created_index) += rxn_deriv;
        }
      }
    }
  }



  template<typename Real>
  void
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_contribution_to_jacobian(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::SparseMatrix<Real, Eigen::RowMajor> &jacobi)
  {
    assert(B_smallest_size < B_largest_size);
    assert(C_smallest_size < C_largest_size);
    assert(B_largest_size <= max_size);
    assert(C_largest_size <= max_size);
    auto index0_B = size_to_index(B_smallest_size);
    auto index1_B = size_to_index(B_largest_size);
    auto index0_C = size_to_index(C_smallest_size);
    auto index1_C = size_to_index(C_largest_size);
    assert(index0_B < index1_B);
    assert(index0_C < index1_C);
    assert(index1_B < x.size());
    assert(index1_C < x.size());

    std::vector<Real> rxn_factors(x.size(), 0.);
    std::vector<Real> rxn_factors_dn(x.size(), 0.);
    for (unsigned int i=index0_B; i<=index1_B; ++i)
    {
      rxn_factors[i] = atoms<Real>(index_to_size(i)) * x(i);
      rxn_factors_dn[i] = atoms<Real>(index_to_size(i));
    }

    if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
    {
      for (unsigned int i=index0_C; i<=index1_C; ++i)
      {
        rxn_factors[i] = atoms<Real>(index_to_size(i)) * x(i);
        rxn_factors_dn[i] = atoms<Real>(index_to_size(i));
      }
    }

    for (unsigned int i = index0_B; i <= index1_B; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(index0_C, i); j <= index1_C; ++j)
      {
        const auto rxn_deriv_i = rate * rxn_factors_dn[i] * rxn_factors[j];
        const auto rxn_deriv_j = rate * rxn_factors[i] * rxn_factors_dn[j];

        jacobi.coeffRef(i, i) -= rxn_deriv_i;
        jacobi.coeffRef(i, j) -= rxn_deriv_j;

        jacobi.coeffRef(j, i) -= rxn_deriv_i;
        jacobi.coeffRef(j, j) -= rxn_deriv_j;

        auto created_size = index_to_size(i) + index_to_size(j);
        if (created_size <= max_size)
        {
          auto created_index = size_to_index(created_size);
          jacobi.coeffRef(created_index, i) += rxn_deriv_i;
          jacobi.coeffRef(created_index, j) += rxn_deriv_j;
        }
      }
    }
  }



  template<typename Real>
  void
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_nonzero_to_jacobian(
      std::vector<Eigen::Triplet<Real>> &triplet_list)
  {
    assert(B_smallest_size < B_largest_size);
    assert(C_smallest_size < C_largest_size);
    assert(B_largest_size <= max_size);
    assert(C_largest_size <= max_size);
    auto index0_B = size_to_index(B_smallest_size);
    auto index1_B = size_to_index(B_largest_size);
    auto index0_C = size_to_index(C_smallest_size);
    auto index1_C = size_to_index(C_largest_size);
    assert(index0_B < index1_B);
    assert(index0_C < index1_C);

    for (unsigned int i = index0_B; i <= index1_B; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(index0_C, i); j <= index1_C; ++j)
      {
        triplet_list.push_back(Eigen::Triplet<Real>(i,i));
        triplet_list.push_back(Eigen::Triplet<Real>(i,j));

        triplet_list.push_back(Eigen::Triplet<Real>(j,i));
        triplet_list.push_back(Eigen::Triplet<Real>(j,j));

        auto created_size = index_to_size(i) + index_to_size(j);
        if (created_size <= max_size)
        {
          auto created_index = size_to_index(created_size);
          triplet_list.push_back(Eigen::Triplet<Real>(created_index,i));
          triplet_list.push_back(Eigen::Triplet<Real>(created_index,j));
        }
      }
    }
  }



  template<typename Real>
  void
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::update_num_nonzero(unsigned int &num_nonzero)
  {
    num_nonzero += (std::max(B_largest_size, C_largest_size) - std::min(B_smallest_size, C_smallest_size))*6;
  }



  ///
  /// A representation of the system of ODEs that describes the nanoparticle formation. A combination of objects
  /// derived from RightHandSideContribution together form a "mechanism" with some chemical interpretation
  ///
  template<typename Real, typename Matrix>
  class Model
  {
  public:
    Model();

    Model(unsigned int nucleation_order, unsigned int max_size);

    void add_rhs_contribution(std::shared_ptr<RightHandSideContribution<Real, Matrix>> &rhs);
    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;
    Matrix jacobian(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;

    unsigned int nucleation_order;
    unsigned int max_size;

  private:
    std::vector<std::shared_ptr<RightHandSideContribution<Real, Matrix>>> rhs_contributions;
  };



  /// Partial specialization for a dense matrix
  template<typename Real>
  class Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
  {
  public:
    Model();

    Model(unsigned int nucleation_order, unsigned int max_size);

    void add_rhs_contribution(
        std::shared_ptr<RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>> &rhs);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> jacobian(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;

    unsigned int nucleation_order;
    unsigned int max_size;

  private:
    std::vector<std::shared_ptr<
        RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>> rhs_contributions;
  };



  template<typename Real>
  Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Model()
    : Model(std::numeric_limits<unsigned int>::signaling_NaN(), std::numeric_limits<unsigned int>::signaling_NaN())
  {}



  template<typename Real>
  Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Model(
      unsigned int nucleation_order, unsigned int max_size)
      : nucleation_order(nucleation_order), max_size(max_size)
  {}



  template<typename Real>
  void Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_rhs_contribution(
      std::shared_ptr<RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>> &rhs)
  {
    rhs_contributions.push_back(rhs);
  }



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::rhs(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    // Initialize the right hand side as a zero vector.
    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x.rows());

    // Loop through every right hand side contribution added to the model and keep adding to the right hand side.
    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->add_contribution_to_rhs(x, rhs);
    }

    return rhs;
  }



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>
  Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::jacobian(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    // initialize the Jacobian as a zero matrix.
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> J
      = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Zero(x.rows(), x.rows());

    // Loop through every right hand side contribution added to the model and keep adding to the Jacobian.
    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->add_contribution_to_jacobian(x, J);
    }

    return J;
  }



  /// Partial specialization for a sparse matrix (row major)
  template<typename Real>
  class Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
  {
  public:
    Model();

    Model(unsigned int nucleation_order, unsigned int max_size);

    void add_rhs_contribution(
        std::shared_ptr<RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>> &rhs);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;
    Eigen::SparseMatrix<Real, Eigen::RowMajor> jacobian(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;

    unsigned int nucleation_order;
    unsigned int max_size;

  private:
    std::vector<std::shared_ptr<
        RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>>> rhs_contributions;
  };



  template<typename Real>
  Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Model()
      : Model(std::numeric_limits<unsigned int>::signaling_NaN(), std::numeric_limits<unsigned int>::signaling_NaN())
  {}



  template<typename Real>
  Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Model(unsigned int nucleation_order, unsigned int max_size)
      : nucleation_order(nucleation_order), max_size(max_size)
  {}



  template<typename Real>
  void Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_rhs_contribution(
      std::shared_ptr<RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>> &rhs)
  {
    rhs_contributions.push_back(rhs);
  }



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::rhs(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    // Initialize the right hand side as a zero vector.
    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x.rows());

    // Loop through every right hand side contribution added to the model and keep adding to the right hand side.
    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->add_contribution_to_rhs(x, rhs);
    }

    return rhs;
  }



  template<typename Real>
  Eigen::SparseMatrix<Real, Eigen::RowMajor>
  Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::jacobian(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    // Form the sparsity pattern by mimicking construction of the matrix, but whenever
    // a value would be calculated, simply add those indices to a list indicating nonzero entries.
    std::vector< Eigen::Triplet<Real> > triplet_list;
    unsigned int estimate_nonzero = 0;

    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->update_num_nonzero(estimate_nonzero);
    }

    triplet_list.reserve(estimate_nonzero);

    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->add_nonzero_to_jacobian(triplet_list);
    }

    Eigen::SparseMatrix<Real, Eigen::RowMajor> J(x.rows(), x.rows());

    J.setFromTriplets(triplet_list.begin(), triplet_list.end());

    // Loop through every right hand side contribution added to the model and keep adding to the Jacobian.
    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->add_contribution_to_jacobian(x, J);
    }

    J.makeCompressed();

    return J;
  }



  /// Partial specialization for a sparse matrix (column major)
  template<typename Real>
  class Model<Real, Eigen::SparseMatrix<Real>>
  {
  public:
    Model();

    Model(unsigned int nucleation_order, unsigned int max_size);

    void add_rhs_contribution(
        std::shared_ptr<RightHandSideContribution<Real, Eigen::SparseMatrix<Real>>> &rhs);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;
    Eigen::SparseMatrix<Real> jacobian(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;

    unsigned int nucleation_order;
    unsigned int max_size;

  private:
    std::vector<std::shared_ptr<
        RightHandSideContribution<Real, Eigen::SparseMatrix<Real>>>> rhs_contributions;
  };



  template<typename Real>
  Model<Real, Eigen::SparseMatrix<Real>>::Model()
      : Model(std::numeric_limits<unsigned int>::signaling_NaN(), std::numeric_limits<unsigned int>::signaling_NaN())
  {}



  template<typename Real>
  Model<Real, Eigen::SparseMatrix<Real>>::Model(unsigned int nucleation_order, unsigned int max_size)
      : nucleation_order(nucleation_order), max_size(max_size)
  {}



  template<typename Real>
  void Model<Real, Eigen::SparseMatrix<Real>>::add_rhs_contribution(
      std::shared_ptr<RightHandSideContribution<Real, Eigen::SparseMatrix<Real>>> &rhs)
  {
    rhs_contributions.push_back(rhs);
  }



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  Model<Real, Eigen::SparseMatrix<Real>>::rhs(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    // Initialize the right hand side as a zero vector.
    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x.rows());

    // Loop through every right hand side contribution added to the model and keep adding to the right hand side.
    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->add_contribution_to_rhs(x, rhs);
    }

    return rhs;
  }



  template<typename Real>
  Eigen::SparseMatrix<Real>
  Model<Real, Eigen::SparseMatrix<Real>>::jacobian(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    // Form the sparsity pattern by mimicking construction of the matrix, but whenever
    // a value would be calculated, simply add those indices to a list indicating nonzero entries.
    std::vector< Eigen::Triplet<Real> > triplet_list;
    unsigned int estimate_nonzero = 0;

    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->update_num_nonzero(estimate_nonzero);
    }

    triplet_list.reserve(estimate_nonzero);

    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->add_nonzero_to_jacobian(triplet_list);
    }

    Eigen::SparseMatrix<Real> J(x.rows(), x.rows());

    J.setFromTriplets(triplet_list.begin(), triplet_list.end());

    // Loop through every right hand side contribution added to the model and keep adding to the Jacobian.
    for (auto & rhs_contribution : rhs_contributions)
    {
      rhs_contribution->add_contribution_to_jacobian(x, J);
    }

    J.makeCompressed();

    return J;
  }

}

#endif //MEPBM_MODELS_H
