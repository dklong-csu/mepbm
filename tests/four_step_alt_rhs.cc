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
  Model::TermolecularNucleation::Parameters prm_nuc(A_index, As_index, POM_index,nucleation_index,
                                                    kf, kb, k1, solvent);
  Model::TermolecularNucleation nucleation;

  // Small Growth
  Model::Growth::Parameters prm_small_growth(A_index, nucleation_order, cutoff, max_size,
                                             POM_index, conserved_size, k2);
  Model::Growth small_growth;

  // Large Growth
  Model::Growth::Parameters prm_large_growth(A_index, cutoff+1, max_size, max_size,
                                             POM_index, conserved_size, k3);
  Model::Growth large_growth;

  // Agglomeration
  Model::Agglomeration::Parameters prm_agglom(nucleation_index, cutoff,
                                              nucleation_index, cutoff,
                                              max_size, conserved_size, k4);
  Model::Agglomeration agglomeration;
  // Create Model
  Model::Model four_step_alt(nucleation_order, max_size);
  four_step_alt.add_rhs_contribution(nucleation, &prm_nuc);
  four_step_alt.add_rhs_contribution(small_growth, &prm_small_growth);
  four_step_alt.add_rhs_contribution(large_growth, &prm_large_growth);
  four_step_alt.add_rhs_contribution(agglomeration, &prm_agglom);

  // Output right hand side
  StateVector state = { 1., .9, .8, .7, .6, .5, .4, .3, .2, .1, .05, .025};
  StateVector rhs(max_size+1, 0.);
  four_step_alt(state, rhs, 0.);

  for (auto val : rhs)
  {
    std::cout << val << std::endl;
  }

}