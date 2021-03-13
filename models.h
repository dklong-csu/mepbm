#ifndef MEPBM_MODELS_H
#define MEPBM_MODELS_H


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
  template<typename Real>
  Real atoms(unsigned int &size, unsigned int conserved_size);


  template<typename Real>
  Real atoms(unsigned int &size,
             unsigned int conserved_size)
  {
    return 2.677 * size * std::pow(1.*conserved_size*size, -0.28);
  }



  /*
 * A base class for describing the effects nucleation, growth, and agglomeration have on the
 * system of ODEs.
 */
  template<typename Real, typename Matrix>
  class RightHandSideContribution
  {
  public:
    virtual void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                         Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) = 0;

    virtual void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                              Matrix &J) = 0;

    virtual void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) = 0;

    virtual void update_num_nonzero(unsigned int &num_nonzero) = 0;

  };



  /*
 * A class which describes the effect of termolecular nucleation on the right hand side of the ODE system.
 * This chemical process is described as
 *        A + 2S <-> A_s + Ligand   (1)
 *        2A_s + A -> B + Ligand    (2)
 * (1) is called the dissapative step.
 * (2) is called nucleation.
 *
 * A = precursor particle
 * S = solvent -- assumed to be a much larger concentration than the other components, and thus constant
 * A_s = disassociated precursor
 * Ligand = ligand (interferes with the precursor)
 * B = particle -- this represents that 2A_s + A describes nucleation, so the smallest possible particle is formed
 *
 * In order to fully define a growth process, this class requires:
 * A_index -- in order to know the location of the precursor within the state vector.
 * As_index -- in order to know the location of the disassociated precursor within the state vector.
 * ligand_index -- in order to know the location of the ligand within the state vector.
 * particle_index -- in order to know the location of the nucleated particle within the state vector.
 * rate_forward -- the rate at which the forward direction of (1) occurs.
 * rate_backward -- the rate at which the backward direction of (1) occurs.
 * rate_nucleation -- the rate at which (2) occurs.
 * solvent -- the amount of solvent present during the entire reaction.
 *
 * This information is housed in the Parameters class and is passed to the right-hand side function.
 */
  template<typename Real, typename Matrix>
  class TermolecularNucleation : public Model::RightHandSideContribution<Real, Matrix>
  {
  public:
    const unsigned int A_index, As_index, ligand_index, particle_index;
    const Real rate_forward, rate_backward, rate_nucleation;
    const Real solvent;

    // default constructor creates an invalid object
    TermolecularNucleation();

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

  /*
   * ==================================================================================================================
   * Specialization for a dense matrix
   * ==================================================================================================================
   */

  template<typename Real>
  class TermolecularNucleation<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
      : public Model::RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
  {
  public:
    const unsigned int A_index, As_index, ligand_index, particle_index;
    const Real rate_forward, rate_backward, rate_nucleation;
    const Real solvent;

    // default constructor creates an invalid object
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



  /*
   * ==================================================================================================================
   * Specialization for a sparse matrix
   * ==================================================================================================================
   */
  template<typename Real>
  class TermolecularNucleation<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
      : public Model::RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
  {
  public:
    const unsigned int A_index, As_index, ligand_index, particle_index;
    const Real rate_forward, rate_backward, rate_nucleation;
    const Real solvent;

    // default constructor creates an invalid object
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



  /*
 * A class which describes the effect of particle growth on the right hand side of the ODE system.
 * This chemical process is described as
 *        A + B -> C + Ligand
 * A = precursor -- think of this as the building block of a particle, but not yet an actual particle
 * B = particle -- these particles are only within some defined range
 * C = larger particle -- this simply symbolizes that a B particle gets bigger when it binds with an A
 * Ligand = a ligand -- this is a molecule which interferes with precursors
 *
 * In order to fully define a growth process, this class requires:
 * A_index -- in order to know the location of the precursor within the state vector.
 * smallest_size -- this is the smallest particle size that B can take.
 * largest_size -- this is the largest particle size that B can take.
 * max_size -- this is the largest particle size we track, particles can grow beyond this size, but are untracked.
 * ligand_index -- in order to know the location of the ligand within the state vector.
 * conserved_size -- the size of the tracked molecule (e.g. dimer -> conserved_size = 2).
 * rate -- the rate constant which describes the speed at which this reaction occurs.
 *
 * This information is housed in the Parameters class and is passed to the right-hand side function.
 */
  template<typename Real, typename Matrix>
  class Growth : public RightHandSideContribution<Real, Matrix>
  {
  public:
    const unsigned int A_index, smallest_size, largest_size, max_size, ligand_index, conserved_size;
    const Real rate;

    // default constructor creates an invalid object
    Growth();

    Growth(unsigned int A_index, unsigned int smallest_size, unsigned int largest_size,
           unsigned int max_size, unsigned int ligand_index, unsigned int conserved_size,
           Real rate);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Matrix &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;
  };



  /*
   * ==================================================================================================================
   * Specialization for a dense matrix
   * ==================================================================================================================
   */

  template<typename Real>
  class Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
      : public RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
  {
  public:
    const unsigned int A_index, smallest_size, largest_size, max_size, ligand_index, conserved_size;
    const Real rate;

    // default constructor creates an invalid object
    Growth();

    Growth(unsigned int A_index, unsigned int smallest_size, unsigned int largest_size,
           unsigned int max_size, unsigned int ligand_index, unsigned int conserved_size,
           Real rate);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) {}

    void update_num_nonzero(unsigned int &num_nonzero) {}
  };



  template<typename Real>
  Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Growth()
      : Growth(std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<Real>::signaling_NaN())
  {}



  template<typename Real>
  Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Growth(
      const unsigned int A_index,
      const unsigned int smallest_size,
      const unsigned int largest_size,
      const unsigned int max_size,
      const unsigned int ligand_index,
      const unsigned int conserved_size,
      const Real rate)
      : A_index(A_index)
      , smallest_size(smallest_size)
      , largest_size(largest_size)
      , max_size(max_size)
      , ligand_index(ligand_index)
      , conserved_size(conserved_size)
      , rate(rate)
  {}



  template<typename Real>
  void Growth<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    // FIXME: turn these into error messages?
    assert(smallest_size < largest_size);
    assert(largest_size <= max_size);
    // FIXME: I should pass some size_to_index function as an argument as this only works if
    // FIXME: the nucleation order is 3 at the moment
    for (unsigned int size = smallest_size; size <= largest_size; ++size)
    {
      const Real rxn_factor = rate *  x(A_index) * atoms<Real>(size, conserved_size) * x(size);
      rhs(size) -= rxn_factor;
      rhs(ligand_index) += rxn_factor;
      rhs(A_index) -= rxn_factor;

      if (size < max_size)
      {
        rhs(size + 1) += rxn_factor;
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

    for (unsigned int size = smallest_size; size <= largest_size; ++size)
    {
      const Real rxn_factor_dA = rate * atoms<Real>(size, conserved_size) * x(size);
      const Real rxn_factor_dn = rate * x(A_index) * atoms<Real>(size, conserved_size);

      jacobi(size, A_index) -= rxn_factor_dA;
      jacobi(size, size) -= rxn_factor_dn;

      jacobi(ligand_index, A_index) += rxn_factor_dA;
      jacobi(ligand_index, size) += rxn_factor_dn;

      jacobi(A_index, A_index) -= rxn_factor_dA;
      jacobi(A_index, size) -= rxn_factor_dn;

      if (size < max_size)
      {
        jacobi(size + 1, A_index) += rxn_factor_dA;
        jacobi(size + 1, size) += rxn_factor_dn;
      }
    }
  }



  /*
   * ==================================================================================================================
   * Specialization for a sparse matrix
   * ==================================================================================================================
   */
  template<typename Real>
  class Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
      : public RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
  {
  public:
    const unsigned int A_index, smallest_size, largest_size, max_size, ligand_index, conserved_size;
    const Real rate;

    // default constructor creates an invalid object
    Growth();

    Growth(unsigned int A_index, unsigned int smallest_size, unsigned int largest_size,
           unsigned int max_size, unsigned int ligand_index, unsigned int conserved_size,
           Real rate);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::SparseMatrix<Real, Eigen::RowMajor> &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;
  };



  template<typename Real>
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Growth()
      : Growth(std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<unsigned int>::signaling_NaN(),
               std::numeric_limits<Real>::signaling_NaN())
  {}



  template<typename Real>
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Growth(const unsigned int A_index,
                                                  const unsigned int smallest_size,
                                                  const unsigned int largest_size,
                                                  const unsigned int max_size,
                                                  const unsigned int ligand_index,
                                                  const unsigned int conserved_size,
                                                  const Real rate)
      : A_index(A_index)
      , smallest_size(smallest_size)
      , largest_size(largest_size)
      , max_size(max_size)
      , ligand_index(ligand_index)
      , conserved_size(conserved_size)
      , rate(rate)
  {}



  template<typename Real>
  void Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    // FIXME: turn these into error messages?
    assert(smallest_size < largest_size);
    assert(largest_size <= max_size);
    // FIXME: I should pass some size_to_index function as an argument as this only works if
    // FIXME: the nucleation order is 3 at the moment
    for (unsigned int size = smallest_size; size <= largest_size; ++size)
    {
      const Real rxn_factor = rate *  x(A_index) * atoms<Real>(size, conserved_size) * x(size);
      rhs(size) -= rxn_factor;
      rhs(ligand_index) += rxn_factor;
      rhs(A_index) -= rxn_factor;

      if (size < max_size)
      {
        rhs(size + 1) += rxn_factor;
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

    for (unsigned int size = smallest_size; size <= largest_size; ++size)
    {
      const Real rxn_factor_dA = rate * atoms<Real>(size, conserved_size) * x(size);
      const Real rxn_factor_dn = rate * x(A_index) * atoms<Real>(size, conserved_size);

      jacobi.coeffRef(size, A_index) -= rxn_factor_dA;
      jacobi.coeffRef(size, size) -= rxn_factor_dn;

      jacobi.coeffRef(ligand_index, A_index) += rxn_factor_dA;
      jacobi.coeffRef(ligand_index, size) += rxn_factor_dn;

      jacobi.coeffRef(A_index, A_index) -= rxn_factor_dA;
      jacobi.coeffRef(A_index, size) -= rxn_factor_dn;

      if (size < max_size)
      {
        jacobi.coeffRef(size + 1, A_index) += rxn_factor_dA;
        jacobi.coeffRef(size + 1, size) += rxn_factor_dn;
      }
    }
  }



  template<typename Real>
  void
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list)
  {
    assert(smallest_size < largest_size);
    assert(largest_size <= max_size);

    for (unsigned int size = smallest_size; size <= largest_size; ++size)
    {
      triplet_list.push_back(Eigen::Triplet<Real>(size, A_index));
      triplet_list.push_back(Eigen::Triplet<Real>(size, size));

      triplet_list.push_back(Eigen::Triplet<Real>(ligand_index, A_index));
      triplet_list.push_back(Eigen::Triplet<Real>(ligand_index, size));

      triplet_list.push_back(Eigen::Triplet<Real>(A_index, A_index));
      triplet_list.push_back(Eigen::Triplet<Real>(A_index, size));

      if (size < max_size)
      {
        triplet_list.push_back(Eigen::Triplet<Real>(size + 1, A_index));
        triplet_list.push_back(Eigen::Triplet<Real>(size + 1, size));
      }
    }
  }



  template<typename Real>
  void
  Growth<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::update_num_nonzero(unsigned int &num_nonzero)
  {
    num_nonzero += (largest_size - smallest_size) * 8;
  }



  /*
 * A class which describes the effect of particle agglomeration on the right hand side of the ODE system.
 * This chemical process is described as
 *        B + C -> D
 * B = particle within some size range
 * C = particle within some (possibly different than B) size range
 * D = larger particle -- this simply symbolizes that a B particle gets bigger when it binds with a C
 *
 * In order to fully define a growth process, this class requires:
 * B_smallest_size -- the smallest particle size a B particle can be.
 * B_largest_size -- the largest particle size a B particle can be.
 * C_smallest_size -- the smallest particle size a C particle can be.
 * C_largest_size -- the largest particle size a C particle can be.
 * max_size -- this is the largest particle size we track, particles can grow beyond this size, but are untracked.
 * conserved_size -- the size of the tracked molecule (e.g. dimer -> conserved_size = 2).
 * rate -- the rate constant which describes the speed at which this reaction occurs.
 *
 * This information is housed in the Parameters class and is passed to the right-hand side function.
 */
  template<typename Real, typename Matrix>
  class Agglomeration : public RightHandSideContribution<Real, Matrix>
  {
  public:
    const unsigned int B_smallest_size, B_largest_size;
    const unsigned int C_smallest_size, C_largest_size;
    const unsigned int max_size;
    const unsigned int conserved_size;
    const Real rate;

    // default constructor creates an invalid object
    Agglomeration();

    Agglomeration(unsigned int B_smallest_size, unsigned int B_largest_size,
                  unsigned int C_smallest_size, unsigned int C_largest_size,
                  unsigned int max_size, unsigned int conserved_size, Real rate);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Matrix &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;
  };



  /*
   * ==================================================================================================================
   * Specialization for a dense matrix
   * ==================================================================================================================
   */

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

    // default constructor creates an invalid object
    Agglomeration();

    Agglomeration(unsigned int B_smallest_size, unsigned int B_largest_size,
                  unsigned int C_smallest_size, unsigned int C_largest_size,
                  unsigned int max_size, unsigned int conserved_size, Real rate);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) {}

    void update_num_nonzero(unsigned int &num_nonzero) {}
  };



  template<typename Real>
  Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Agglomeration()
      : Agglomeration(std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<Real>::signaling_NaN())
  {}



  template<typename Real>
  Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::Agglomeration(
      const unsigned int B_smallest_size,
      const unsigned int B_largest_size,
      const unsigned int C_smallest_size,
      const unsigned int C_largest_size,
      const unsigned int max_size,
      const unsigned int conserved_size,
      const Real rate)
      : B_smallest_size(B_smallest_size), B_largest_size(B_largest_size)
      , C_smallest_size(C_smallest_size), C_largest_size(C_largest_size)
      , max_size(max_size)
      , conserved_size(conserved_size)
      , rate(rate)
  {}



  template<typename Real>
  void Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    // FIXME: turn these into error messages?
    assert(B_smallest_size < B_largest_size);
    assert(C_smallest_size < C_largest_size);
    assert(B_largest_size <= max_size);
    assert(C_largest_size <= max_size);

    // Pre-calculate terms of the derivative that will be used since we require an expensive double loop.
    // Many of the terms will be calculated more than once if they are not pre-calculated, which adds
    // up as the double loop gets larger.
    //
    // The factor is calculated to be the number of binding sites * current concentration
    std::vector<Real> rxn_factors(x.size(), 0.);
    for (unsigned int i=B_smallest_size; i<=B_largest_size; ++i)
    {
      rxn_factors[i] = atoms<Real>(i, conserved_size) * x(i);
    }

    // FIXME: is this even necessary to do? It might be simpler to just calculate from
    // FIXME: min(B_smallest_size, C_smallest_size) to max(B_largest_size, C_largest_size)
    if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
    {
      for (unsigned int i=C_smallest_size; i<=C_largest_size; ++i)
      {
        rxn_factors[i] = atoms<Real>(i, conserved_size) * x(i);
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
    for (unsigned int i = B_smallest_size; i <= B_largest_size; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(C_smallest_size, i); j <= C_largest_size; ++j)
      {
        const auto rxn_deriv = rate * rxn_factors[i] * rxn_factors[j];
        rhs(i) -= rxn_deriv;
        rhs(j) -= rxn_deriv;
        if (i+j <= max_size)
          rhs(i+j) += rxn_deriv;
      }
    }
  }



  template<typename Real>
  void
  Agglomeration<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::add_contribution_to_jacobian(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real,Eigen::Dynamic, Eigen::Dynamic> &jacobi)
  {
    std::vector<Real> rxn_factors(x.size(), 0.);
    std::vector<Real> rxn_factors_dn(x.size(), 0.);
    for (unsigned int i=B_smallest_size; i<=B_largest_size; ++i)
    {
      rxn_factors[i] = atoms<Real>(i, conserved_size) * x(i);
      rxn_factors_dn[i] = atoms<Real>(i, conserved_size);
    }

    // FIXME: is this even necessary to do? It might be simpler to just calculate from
    // FIXME: min(B_smallest_size, C_smallest_size) to max(B_largest_size, C_largest_size)
    if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
    {
      for (unsigned int i=C_smallest_size; i<=C_largest_size; ++i)
      {
        rxn_factors[i] = atoms<Real>(i, conserved_size) * x(i);
        rxn_factors_dn[i] = atoms<Real>(i, conserved_size);
      }
    }

    for (unsigned int i = B_smallest_size; i <= B_largest_size; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(C_smallest_size, i); j <= C_largest_size; ++j)
      {
        const auto rxn_deriv_i = rate * rxn_factors_dn[i] * rxn_factors[j];
        const auto rxn_deriv_j = rate * rxn_factors[i] * rxn_factors_dn[j];

        jacobi(i, i) -= rxn_deriv_i;
        jacobi(i, j) -= rxn_deriv_j;

        jacobi(j, i) -= rxn_deriv_i;
        jacobi(j, j) -= rxn_deriv_j;

        if (i+j <= max_size)
        {
          jacobi(i+j, i) += rxn_deriv_i;
          jacobi(i+j, j) += rxn_deriv_j;
        }
      }
    }
  }



  /*
   * ==================================================================================================================
   * Specialization for a sparse matrix
   * ==================================================================================================================
   */

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

    // default constructor creates an invalid object
    Agglomeration();

    Agglomeration(unsigned int B_smallest_size, unsigned int B_largest_size,
                  unsigned int C_smallest_size, unsigned int C_largest_size,
                  unsigned int max_size, unsigned int conserved_size, Real rate);

    void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                 Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs) override;

    void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
                                      Eigen::SparseMatrix<Real, Eigen::RowMajor> &jacobi) override;

    void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) override;

    void update_num_nonzero(unsigned int &num_nonzero) override;
  };



  template<typename Real>
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Agglomeration()
      : Agglomeration(std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<unsigned int>::signaling_NaN(),
                      std::numeric_limits<Real>::signaling_NaN())
  {}



  template<typename Real>
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::Agglomeration(
      const unsigned int B_smallest_size,
      const unsigned int B_largest_size,
      const unsigned int C_smallest_size,
      const unsigned int C_largest_size,
      const unsigned int max_size,
      const unsigned int conserved_size,
      const Real rate)
      : B_smallest_size(B_smallest_size), B_largest_size(B_largest_size)
      , C_smallest_size(C_smallest_size), C_largest_size(C_largest_size)
      , max_size(max_size)
      , conserved_size(conserved_size)
      , rate(rate)
  {}



  template<typename Real>
  void Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_contribution_to_rhs(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    // FIXME: turn these into error messages?
    assert(B_smallest_size < B_largest_size);
    assert(C_smallest_size < C_largest_size);
    assert(B_largest_size <= max_size);
    assert(C_largest_size <= max_size);

    // Pre-calculate terms of the derivative that will be used since we require an expensive double loop.
    // Many of the terms will be calculated more than once if they are not pre-calculated, which adds
    // up as the double loop gets larger.
    //
    // The factor is calculated to be the number of binding sites * current concentration
    std::vector<Real> rxn_factors(x.size(), 0.);
    for (unsigned int i=B_smallest_size; i<=B_largest_size; ++i)
    {
      rxn_factors[i] = atoms<Real>(i, conserved_size) * x(i);
    }

    // FIXME: is this even necessary to do? It might be simpler to just calculate from
    // FIXME: min(B_smallest_size, C_smallest_size) to max(B_largest_size, C_largest_size)
    if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
    {
      for (unsigned int i=C_smallest_size; i<=C_largest_size; ++i)
      {
        rxn_factors[i] = atoms<Real>(i, conserved_size) * x(i);
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
    for (unsigned int i = B_smallest_size; i <= B_largest_size; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(C_smallest_size, i); j <= C_largest_size; ++j)
      {
        const auto rxn_deriv = rate * rxn_factors[i] * rxn_factors[j];
        rhs(i) -= rxn_deriv;
        rhs(j) -= rxn_deriv;
        if (i+j <= max_size)
          rhs(i+j) += rxn_deriv;
      }
    }
  }



  template<typename Real>
  void
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_contribution_to_jacobian(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x,
      Eigen::SparseMatrix<Real, Eigen::RowMajor> &jacobi)
  {
    std::vector<Real> rxn_factors(x.size(), 0.);
    std::vector<Real> rxn_factors_dn(x.size(), 0.);
    for (unsigned int i=B_smallest_size; i<=B_largest_size; ++i)
    {
      rxn_factors[i] = atoms<Real>(i, conserved_size) * x(i);
      rxn_factors_dn[i] = atoms<Real>(i, conserved_size);
    }

    // FIXME: is this even necessary to do? It might be simpler to just calculate from
    // FIXME: min(B_smallest_size, C_smallest_size) to max(B_largest_size, C_largest_size)
    if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
    {
      for (unsigned int i=C_smallest_size; i<=C_largest_size; ++i)
      {
        rxn_factors[i] = atoms<Real>(i, conserved_size) * x(i);
        rxn_factors_dn[i] = atoms<Real>(i, conserved_size);
      }
    }

    for (unsigned int i = B_smallest_size; i <= B_largest_size; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(C_smallest_size, i); j <= C_largest_size; ++j)
      {
        const auto rxn_deriv_i = rate * rxn_factors_dn[i] * rxn_factors[j];
        const auto rxn_deriv_j = rate * rxn_factors[i] * rxn_factors_dn[j];

        jacobi.coeffRef(i, i) -= rxn_deriv_i;
        jacobi.coeffRef(i, j) -= rxn_deriv_j;

        jacobi.coeffRef(j, i) -= rxn_deriv_i;
        jacobi.coeffRef(j, j) -= rxn_deriv_j;

        if (i+j <= max_size)
        {
          jacobi.coeffRef(i+j, i) += rxn_deriv_i;
          jacobi.coeffRef(i+j, j) += rxn_deriv_j;
        }
      }
    }
  }



  template<typename Real>
  void
  Agglomeration<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::add_nonzero_to_jacobian(
      std::vector<Eigen::Triplet<Real>> &triplet_list)
  {
    for (unsigned int i = B_smallest_size; i <= B_largest_size; ++i)
    {
      // If the B and C size ranges overlap, then we end up double counting some contributions.
      // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
      for (unsigned int j = std::max(C_smallest_size, i); j <= C_largest_size; ++j)
      {
        triplet_list.push_back(Eigen::Triplet<Real>(i,i));
        triplet_list.push_back(Eigen::Triplet<Real>(i,j));

        triplet_list.push_back(Eigen::Triplet<Real>(j,i));
        triplet_list.push_back(Eigen::Triplet<Real>(j,j));

        if (i+j <= max_size)
        {
          triplet_list.push_back(Eigen::Triplet<Real>(i+j,i));
          triplet_list.push_back(Eigen::Triplet<Real>(i+j,j));
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



  // A model is a representation of the system of ODEs that's being solved. It needs to know
  // the smallest and largest particle size (nucleation_order, max_size). The model also needs
  // a way to evaluate the right-hand side and the Jacobian of the system of ODEs.
  template<typename Real, typename Matrix>
  class Model
  {
  public:
    Model(unsigned int nucleation_order, unsigned int max_size);

    void add_rhs_contribution(std::shared_ptr<RightHandSideContribution<Real, Matrix>> &rhs);
    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;
    Matrix jacobian(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;

    const unsigned int nucleation_order;
    const unsigned int max_size;

  private:
    std::vector<std::shared_ptr<RightHandSideContribution<Real, Matrix>>> rhs_contributions;
  };


  /*
   * ==================================================================================================================
   * Specialization for a dense matrix
   * ==================================================================================================================
   */

  template<typename Real>
  class Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
  {
  public:
    Model(unsigned int nucleation_order, unsigned int max_size);

    void add_rhs_contribution(
        std::shared_ptr<RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>> &rhs);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> jacobian(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;

    const unsigned int nucleation_order;
    const unsigned int max_size;

  private:
    std::vector<std::shared_ptr<
        RightHandSideContribution<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>> rhs_contributions;
  };



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



  /*
   * ==================================================================================================================
   * Specialization for a sparse matrix
   * ==================================================================================================================
   */

  template<typename Real>
  class Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>
  {
  public:
    Model(unsigned int nucleation_order, unsigned int max_size);

    void add_rhs_contribution(
        std::shared_ptr<RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>> &rhs);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;
    Eigen::SparseMatrix<Real, Eigen::RowMajor> jacobian(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const;

    const unsigned int nucleation_order;
    const unsigned int max_size;

  private:
    std::vector<std::shared_ptr<
        RightHandSideContribution<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>>> rhs_contributions;
  };



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

}

#endif //MEPBM_MODELS_H
