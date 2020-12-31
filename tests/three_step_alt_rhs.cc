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
  const unsigned int max_size = 6;
  const unsigned int nucleation_order = 3;
  const double solvent = 2.;
  const unsigned int conserved_size = 1;

  const unsigned int A_index = 0;
  const unsigned int As_index = 1;
  const unsigned int POM_index = 2;
  const unsigned int nucleation_index = 3;

  const double kf = 100.;
  const double kb = 90.;
  const double k1 = 80.;
  const double k2 = 70.;
  const double k3 = 60.;
  const double cutoff = 4.;


  // Nucleation
  std::shared_ptr<Model::RightHandSideContribution> nucleation
    = std::make_shared<Model::TermolecularNucleation>(A_index, As_index, POM_index,nucleation_index,
                                                      kf, kb, k1, solvent);

  // Small Growth
  std::shared_ptr<Model::RightHandSideContribution> small_growth
    = std::make_shared<Model::Growth>(A_index, nucleation_order, cutoff, max_size,
                                      POM_index, conserved_size, k2);

  // Large Growth
  std::shared_ptr<Model::RightHandSideContribution> large_growth
    = std::make_shared<Model::Growth>(A_index, cutoff+1, max_size, max_size,
                                      POM_index, conserved_size, k3);

  // Create Model
  Model::Model three_step_alt(nucleation_order, max_size);
  three_step_alt.add_rhs_contribution(nucleation);
  three_step_alt.add_rhs_contribution(small_growth);
  three_step_alt.add_rhs_contribution(large_growth);

  // Output right hand side
  StateVector state(max_size+1);
  state(0) = 1.;
  state(1) = .9;
  state(2) = .8;
  state(3) = .7;
  state(4) = .6;
  state(5) = .5;
  state(6) = .4;

  StateVector rhs = three_step_alt.rhs(state);

  std::cout << rhs << std::endl;

}
