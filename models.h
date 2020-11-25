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
  double atoms(unsigned int &size, unsigned int conserved_size);



  /*
 * A base class which describes the necessary parameters to accurately describe the contribution
 * to the right hand side of the ODE system that a certain phase of the nucleation-growth-agglomeration
 * process has.
 */
  class ParametersBase
  {
  public:
    virtual ~ParametersBase() = default;
  };



  /*
 * A base class for describing the effects nucleation, growth, and agglomeration have on the
 * system of ODEs.
 */
  class RightHandSideContribution
  {
  public:
    virtual void add_contribution_to_rhs(const std::vector<double> &x,
                                         std::vector<double> &rhs,
                                         ParametersBase *parameters) = 0;
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
  class TermolecularNucleation : public Model::RightHandSideContribution
  {
  public:
    class Parameters : public Model::ParametersBase
    {
    public:
      const unsigned int A_index, As_index, ligand_index, particle_index;
      const double rate_forward, rate_backward, rate_nucleation;
      const double solvent;


      Parameters(unsigned int A_index, unsigned int As_index, unsigned int ligand_index,
                 unsigned int particle_index,
                 double rate_forward, double rate_backward, double rate_nucleation,
                 double solvent);
    };


    void add_contribution_to_rhs(const std::vector<double> &x,
                                 std::vector<double> &rhs,
                                 Model::ParametersBase *parameters) override;
  };



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
  class Growth : public RightHandSideContribution
  {
  public:
    class Parameters : public ParametersBase
    {
    public:
      const unsigned int A_index, smallest_size, largest_size, max_size, ligand_index, conserved_size;
      const double rate;

      Parameters(unsigned int A_index, unsigned int smallest_size, unsigned int largest_size,
                 unsigned int max_size, unsigned int ligand_index, unsigned int conserved_size,
                 double rate);
    };


    void add_contribution_to_rhs(const std::vector<double> &x,
                                 std::vector<double> &rhs,
                                 ParametersBase *parameters) override;
  };



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
  class Agglomeration : public RightHandSideContribution
  {
  public:
    class Parameters : public ParametersBase
    {
    public:
      const unsigned int B_smallest_size, B_largest_size;
      const unsigned int C_smallest_size, C_largest_size;
      const unsigned int max_size;
      const unsigned int conserved_size;
      const double rate;


      Parameters(unsigned int B_smallest_size, unsigned int B_largest_size,
                 unsigned int C_smallest_size, unsigned int C_largest_size,
                 unsigned int max_size, unsigned int conserved_size, double rate);
    };


    void add_contribution_to_rhs(const std::vector<double> &x,
                                 std::vector<double> &rhs,
                                 ParametersBase *parameters) override;
  };



  // An interface class which creates a rule for forming the right-hand side of a system of ODEs.
  // A series of right-hand side contributions are added to this object via objects derived from
  // the RightHandSideContribution class. The class is able to combine all of these contributions
  // to form a complete right-hand side via the ( ) operator, which is created in a way that
  // interfaces with the ODE solvers in the Boost library.
  class Model
  {
  public:
    Model(unsigned int nucleation_order, unsigned int max_size);

    void add_rhs_contribution(RightHandSideContribution &rhs,
                              ParametersBase * &&prm);

    void operator()(const std::vector<double> &x,
                    std::vector<double> &rhs,
                    double  /* t */);

    const unsigned int nucleation_order;
    const unsigned int max_size;
  private:
    // FIXME: Wolfgang said there is a better way to deal with the pointers here
    std::vector<RightHandSideContribution*> rhs_contributions;
    std::vector<ParametersBase*> contribution_parameters;
  };

}

#endif //MEPBM_MODELS_H
