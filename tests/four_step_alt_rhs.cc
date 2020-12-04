#include <iostream>
#include <vector>
#include <boost/numeric/odeint.hpp>
#include "models.h"

using StateVector = boost::numeric::ublas::vector<double>;

/*
 * This is part of a series of tests to confirm the modular RightHandSide derived classes work as intended.
 * This will test the simplest mechanism, the two-step. This includes a nucleation step and a growth step.
 * We use termolecular nucleation so that we can compare to verified output.
 */

int main()
{
  const unsigned int max_size = 11;
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
  const double k4 = 50.;
  const double cutoff = 5.;


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
                                      POM_index, conserved_size, k3);;

  // Agglomeration
  std::shared_ptr<Model::RightHandSideContribution> agglomeration
    = std::make_shared<Model::Agglomeration>(nucleation_index, cutoff,
                                             nucleation_index, cutoff,
                                             max_size, conserved_size, k4);

  // Create Model
  Model::Model four_step_alt(nucleation_order, max_size);
  four_step_alt.add_rhs_contribution(nucleation);
  four_step_alt.add_rhs_contribution(small_growth);
  four_step_alt.add_rhs_contribution(large_growth);
  four_step_alt.add_rhs_contribution(agglomeration);

  // Output right hand side
  StateVector state(max_size+1);
  state[0] = 1.;
  state[1] = .9;
  state[2] = .8;
  state[3] = .7;
  state[4] = .6;
  state[5] = .5;
  state[6] = .4;
  state[7] = .3;
  state[8] = .2;
  state[9] = .1;
  state[10] = .05;
  state[11] = .025;
  StateVector rhs(max_size+1, 0.);
  four_step_alt(state, rhs, 0.);

  for (auto val : rhs)
  {
    std::cout << val << std::endl;
  }

}
