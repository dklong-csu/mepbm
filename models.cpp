#include "models.h"
#include <cmath>
#include <vector>
#include <cassert>


using StateVector = std::vector<double>;



double Model::atoms(unsigned int &size,
                    unsigned int conserved_size)
{
  const auto new_size = static_cast<double>(size);
  return 2.677 * new_size * std::pow(conserved_size*new_size, -0.28);
}



Model::TermolecularNucleation::Parameters::Parameters(const unsigned int A_index,
                                                      const unsigned int As_index,
                                                      const unsigned int ligand_index,
                                                      const unsigned int particle_index,
                                                      const double rate_forward,
                                                      const double rate_backward,
                                                      const double rate_nucleation,
                                                      const double solvent)
    : A_index(A_index), As_index(As_index), ligand_index(ligand_index), particle_index(particle_index)
    , rate_forward(rate_forward), rate_backward(rate_backward), rate_nucleation(rate_nucleation)
    , solvent(solvent)
{}



void Model::TermolecularNucleation::add_contribution_to_rhs(const StateVector &x, StateVector &rhs,
                                                            ParametersBase *parameters)
{
  const auto& prm = dynamic_cast<const TermolecularNucleation::Parameters*>(parameters);

  const double diss_forward = prm->rate_forward * x[prm->A_index] * prm->solvent*prm->solvent;
  const double diss_backward = prm->rate_backward * x[prm->As_index] * x[prm->ligand_index];
  const double nucleation = prm->rate_nucleation * x[prm->A_index] * x[prm->As_index] * x[prm->As_index];

  rhs[prm->A_index] += -diss_forward + diss_backward - nucleation;
  rhs[prm->As_index] += diss_forward - diss_backward - 2 * nucleation;
  rhs[prm->ligand_index] += diss_forward - diss_backward + nucleation;
  rhs[prm->particle_index] += nucleation;
}



Model::Growth::Parameters::Parameters(const unsigned int A_index,
                                      const unsigned int smallest_size,
                                      const unsigned int largest_size,
                                      const unsigned int max_size,
                                      const unsigned int ligand_index,
                                      const unsigned int conserved_size,
                                      const double rate)
    : A_index(A_index)
    , smallest_size(smallest_size)
    , largest_size(largest_size)
    , max_size(max_size)
    , ligand_index(ligand_index)
    , conserved_size(conserved_size)
    , rate(rate)
{}



void Model::Growth::add_contribution_to_rhs(const StateVector &x,
                                            StateVector &rhs,
                                            ParametersBase *parameters)
{
  const auto& prm = dynamic_cast<const Growth::Parameters*>(parameters);

  // FIXME: turn these into error messages?
  assert(prm->smallest_size < prm->largest_size);
  assert(prm->largest_size <= prm->max_size);
  // FIXME: I should pass some size_to_index function as an argument as this only works if
  // FIXME: the nucleation order is 3 at the moment
  for (unsigned int size = prm->smallest_size; size <= prm->largest_size; ++size)
  {
    const double rxn_factor = prm->rate *  x[prm->A_index] * atoms(size, prm->conserved_size) * x[size];
    rhs[size] -= rxn_factor;
    rhs[prm->ligand_index] += rxn_factor;
    rhs[prm->A_index] -= rxn_factor;

    if (size < prm->max_size)
    {
      rhs[size + 1] += rxn_factor;
    }
  }
}



Model::Agglomeration::Parameters::Parameters(const unsigned int B_smallest_size,
                                             const unsigned int B_largest_size,
                                             const unsigned int C_smallest_size,
                                             const unsigned int C_largest_size,
                                             const unsigned int max_size,
                                             const unsigned int conserved_size,
                                             const double rate)
    : B_smallest_size(B_smallest_size), B_largest_size(B_largest_size)
    , C_smallest_size(C_smallest_size), C_largest_size(C_largest_size)
    , max_size(max_size)
    , conserved_size(conserved_size)
    , rate(rate)
{}



void Model::Agglomeration::add_contribution_to_rhs(const StateVector &x, StateVector &rhs,
                                                   ParametersBase *parameters)
{
  const auto& prm = dynamic_cast<const Agglomeration::Parameters*>(parameters);

  // FIXME: turn these into error messages?
  assert(prm->B_smallest_size < prm->B_largest_size);
  assert(prm->C_smallest_size < prm->C_largest_size);
  assert(prm->B_largest_size <= prm->max_size);
  assert(prm->C_largest_size <= prm->max_size);

  // Pre-calculate terms of the derivative that will be used since we require an expensive double loop.
  // Many of the terms will be calculated more than once if they are not pre-calculated, which adds
  // up as the double loop gets larger.
  //
  // The factor is calculated to be the number of binding sites * current concentration
  std::vector<double> rxn_factors(x.size(), 0.);
  for (unsigned int i=prm->B_smallest_size; i<=prm->B_largest_size; ++i)
  {
    rxn_factors[i] = atoms(i, prm->conserved_size) * x[i];
  }

  // FIXME: is this even necessary to do? It might be simpler to just calculate from
  // FIXME: min(B_smallest_size, C_smallest_size) to max(B_largest_size, C_largest_size)
  if (prm->B_smallest_size != prm->C_smallest_size || prm->B_largest_size != prm->C_smallest_size)
  {
    for (unsigned int i=prm->C_smallest_size; i<=prm->C_largest_size; ++i)
    {
      rxn_factors[i] = atoms(i, prm->conserved_size) * x[i];
    }
  }

  // Calculate the right-hand side contributions from agglomeration
  // Two particles interact with each other, one B particle and one C particle.
  // The rate this occurs based on how frequently a B particle and C particle will interact
  // with each other as well as by how quickly the reaction occurs.
  // The frequency of the particles meeting is proportional to the product of the rxn_factors calculated above.
  // Then the provided rate constant, prm->rate, scales the product to appropriately compute the reaction rate.
  //
  // We always track the outflow due to agglomeration. However, agglomeration might yield a particle whose
  // size is larger than we track. For example, if the max particle size tracked is 10 and agglomeration
  // occurs between a particle of size 6 and 7, then the loss of size 6 and 7 particles will be calculated
  // but the gain of size 13 particles will not because only particles up to size 10 are tracked.
  for (unsigned int i = prm->B_smallest_size; i <= prm->B_largest_size; ++i)
  {
    // If the B and C size ranges overlap, then we end up double counting some contributions.
    // Taking the max between the B-size and the smallest C-size ensures this double counting does not occur.
    for (unsigned int j = std::max(prm->C_smallest_size, i); j <= prm->C_largest_size; ++j)
    {
      const auto rxn_deriv = prm->rate * rxn_factors[i] * rxn_factors[j];
      rhs[i] -= rxn_deriv;
      rhs[j] -= rxn_deriv;
      if (i+j <= prm->max_size)
        rhs[i+j] += rxn_deriv;
    }
  }
}



Model::Model::Model(unsigned int nucleation_order, unsigned int max_size)
    : nucleation_order(nucleation_order), max_size(max_size)
{}



void Model::Model::add_rhs_contribution(RightHandSideContribution &rhs,
                                        ParametersBase * &&prm)
{
  rhs_contributions.push_back(&rhs);
  contribution_parameters.push_back(prm);
}



void Model::Model::operator()(const StateVector &x, StateVector &rhs, double  /* t */)
{
  for (double & rh : rhs)
  {
    rh = 0.;
  }

  for (unsigned int i = 0; i < rhs_contributions.size(); ++i)
  {
    rhs_contributions[i]->add_contribution_to_rhs(x, rhs, contribution_parameters[i]);
  }
}

