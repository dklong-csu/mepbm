#include <iostream>
#include <eigen3/Eigen/Dense>
#include "models.h"

using StateVector = Eigen::VectorXd;

/*
 * This is part of a series of tests to confirm the modular RightHandSide derived classes work as intended.
 * This will test the simplest mechanism, the two-step. This includes a nucleation step and a growth step.
 * We use termolecular nucleation so that we can compare to verified output.
 */

int main()
{
  const unsigned int max_size = 5;
  const unsigned int nucleation_order = 3;
  const double solvent = 2.;
  const unsigned int conserved_size = 1;

  const unsigned int A_index = 0;
  const unsigned int As_index = 1;
  const unsigned int POM_index = 2;
  const unsigned int nucleation_index = 3;

  const double kf = 100.;
  const double kb = 70.;
  const double k1 = 60.;
  const double k2 = 40.;

  // Nucleation
  std::shared_ptr<Model::RightHandSideContribution> nucleation
    = std::make_shared<Model::TermolecularNucleation>(A_index, As_index, POM_index,nucleation_index,
                                                      kf, kb, k1, solvent);

  // Growth
  std::shared_ptr<Model::RightHandSideContribution> growth
    = std::make_shared<Model::Growth>(A_index, nucleation_order, max_size, max_size,
                                      POM_index, conserved_size, k2);

  // Create Model
  Model::Model two_step_alt(nucleation_order, max_size);
  two_step_alt.add_rhs_contribution(nucleation);
  two_step_alt.add_rhs_contribution(growth);

  // Output right hand side
  StateVector state(max_size+1);
  state(0) = 1.;
  state(1) = .8;
  state(2) = .6;
  state(3) = .4;
  state(4) = .2;
  state(5) = .1;

  StateVector rhs = three_step_alt.rhs(state);

  std::cout << rhs << std::endl;

}
