#include <iostream>
#include <vector>
#include <boost/numeric/odeint.hpp>
#include "models.h"

using StateVector = std::vector<double>;

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
  Model::TermolecularNucleation nucleation(A_index, As_index, POM_index,nucleation_index,
                                           kf, kb, k1, solvent);

  // Growth
  Model::Growth growth(A_index, nucleation_order, max_size, max_size,
                       POM_index, conserved_size, k2);

  // Create Model
  Model::Model two_step_alt(nucleation_order, max_size);
  two_step_alt.add_rhs_contribution(nucleation);
  two_step_alt.add_rhs_contribution(growth);

  // Output right hand side
  StateVector state = { 1., .8, .6, .4, .2, .1};
  StateVector rhs(6, 0.);
  two_step_alt(state, rhs, 0.);

  for (auto val : rhs)
  {
    std::cout << val << std::endl;
  }

}