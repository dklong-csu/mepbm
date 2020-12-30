#include "models.h"
#include <cmath>
#include <vector>
#include <cassert>
#include <limits>
#include <eigen3/Eigen/Dense>


using StateVector = Eigen::VectorXd;
using StateMatrix = Eigen::MatrixXd;



double Model::atoms(unsigned int &size,
                    unsigned int conserved_size)
{
  const auto new_size = static_cast<double>(size);
  return 2.677 * new_size * std::pow(conserved_size*new_size, -0.28);
}



Model::TermolecularNucleation::TermolecularNucleation()
: TermolecularNucleation(std::numeric_limits<unsigned int>::signaling_NaN(),
                         std::numeric_limits<unsigned int>::signaling_NaN(),
                         std::numeric_limits<unsigned int>::signaling_NaN(),
                         std::numeric_limits<unsigned int>::signaling_NaN(),
                         std::numeric_limits<double>::signaling_NaN(),
                         std::numeric_limits<double>::signaling_NaN(),
                         std::numeric_limits<double>::signaling_NaN(),
                         std::numeric_limits<double>::signaling_NaN())
{}



Model::TermolecularNucleation::TermolecularNucleation(const unsigned int A_index,
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



void Model::TermolecularNucleation::add_contribution_to_rhs(const StateVector &x, StateVector &rhs)
{
  const double diss_forward = rate_forward * x(A_index) * solvent*solvent;
  const double diss_backward = rate_backward * x(As_index) * x(ligand_index);
  const double nucleation = rate_nucleation * x(A_index) * x(As_index) * x(As_index);

  rhs(A_index) += -diss_forward + diss_backward - nucleation;
  rhs(As_index) += diss_forward - diss_backward - 2 * nucleation;
  rhs(ligand_index) += diss_forward - diss_backward + nucleation;
  rhs(particle_index) += nucleation;
}



void Model::TermolecularNucleation::add_contribution_to_jacobian(const StateVector &x,
                                                                 StateMatrix &jacobi)
{
  const double diss_forward_dA = rate_forward * solvent * solvent;

  const double diss_backward_dAs = rate_backward * x(ligand_index);
  const double diss_backward_dL = rate_backward * x(As_index);

  const double nucleation_dA = rate_nucleation * x(As_index) * x(As_index);
  const double nucleation_dAs = 2 * rate_nucleation * x(A_index) * x(As_index);

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



Model::Growth::Growth()
: Growth(std::numeric_limits<unsigned int>::signaling_NaN(),
         std::numeric_limits<unsigned int>::signaling_NaN(),
         std::numeric_limits<unsigned int>::signaling_NaN(),
         std::numeric_limits<unsigned int>::signaling_NaN(),
         std::numeric_limits<unsigned int>::signaling_NaN(),
         std::numeric_limits<unsigned int>::signaling_NaN(),
         std::numeric_limits<double>::signaling_NaN())
{}



Model::Growth::Growth(const unsigned int A_index,
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
                                            StateVector &rhs)
{
  // FIXME: turn these into error messages?
  assert(smallest_size < largest_size);
  assert(largest_size <= max_size);
  // FIXME: I should pass some size_to_index function as an argument as this only works if
  // FIXME: the nucleation order is 3 at the moment
  for (unsigned int size = smallest_size; size <= largest_size; ++size)
  {
    const double rxn_factor = rate *  x(A_index) * atoms(size, conserved_size) * x(size);
    rhs(size) -= rxn_factor;
    rhs(ligand_index) += rxn_factor;
    rhs(A_index) -= rxn_factor;

    if (size < max_size)
    {
      rhs(size + 1) += rxn_factor;
    }
  }
}



void Model::Growth::add_contribution_to_jacobian(const StateVector &x,
                                                 StateMatrix &jacobi)
{
  assert(smallest_size < largest_size);
  assert(largest_size <= max_size);

  for (unsigned int size = smallest_size; size <= largest_size; ++size)
  {
    const double rxn_factor_dA = rate * atoms(size, conserved_size) * x(size);
    const double rxn_factor_dn = rate * x(A_index) * atoms(size, conserved_size);

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



Model::Agglomeration::Agglomeration()
: Agglomeration(std::numeric_limits<unsigned int>::signaling_NaN(),
                std::numeric_limits<unsigned int>::signaling_NaN(),
                std::numeric_limits<unsigned int>::signaling_NaN(),
                std::numeric_limits<unsigned int>::signaling_NaN(),
                std::numeric_limits<unsigned int>::signaling_NaN(),
                std::numeric_limits<unsigned int>::signaling_NaN(),
                std::numeric_limits<double>::signaling_NaN())
{}



Model::Agglomeration::Agglomeration(const unsigned int B_smallest_size,
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



void Model::Agglomeration::add_contribution_to_rhs(const StateVector &x, StateVector &rhs)
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
  std::vector<double> rxn_factors(x.size(), 0.);
  for (unsigned int i=B_smallest_size; i<=B_largest_size; ++i)
  {
    rxn_factors[i] = atoms(i, conserved_size) * x(i);
  }

  // FIXME: is this even necessary to do? It might be simpler to just calculate from
  // FIXME: min(B_smallest_size, C_smallest_size) to max(B_largest_size, C_largest_size)
  if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
  {
    for (unsigned int i=C_smallest_size; i<=C_largest_size; ++i)
    {
      rxn_factors[i] = atoms(i, conserved_size) * x(i);
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



void Model::Agglomeration::add_contribution_to_jacobian(const StateVector &x,
                                                        StateMatrix &jacobi)
{
  std::vector<double> rxn_factors(x.size(), 0.);
  std::vector<double> rxn_factors_dn(x.size(), 0.);
  for (unsigned int i=B_smallest_size; i<=B_largest_size; ++i)
  {
    rxn_factors[i] = atoms(i, conserved_size) * x(i);
    rxn_factors_dn[i] = atoms(i, conserved_size);
  }

  // FIXME: is this even necessary to do? It might be simpler to just calculate from
  // FIXME: min(B_smallest_size, C_smallest_size) to max(B_largest_size, C_largest_size)
  if (B_smallest_size != C_smallest_size || B_largest_size != C_smallest_size)
  {
    for (unsigned int i=C_smallest_size; i<=C_largest_size; ++i)
    {
      rxn_factors[i] = atoms(i, conserved_size) * x(i);
      rxn_factors_dn[i] = atoms(i, conserved_size);
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



Model::Model::Model(unsigned int nucleation_order, unsigned int max_size)
    : nucleation_order(nucleation_order), max_size(max_size)
{}



void Model::Model::add_rhs_contribution(std::shared_ptr<RightHandSideContribution> &rhs)
{
  rhs_contributions.push_back(rhs);
}



StateVector Model::Model::rhs(StateVector &x)
{
  // Initialize the right hand side as a zero vector.
  StateVector rhs = StateVector::Zero(x.rows());

  // Loop through every right hand side contribution added to the model and keep adding to the right hand side.
  for (auto & rhs_contribution : rhs_contributions)
  {
    rhs_contribution->add_contribution_to_rhs(x, rhs);
  }

  return rhs;
}



StateMatrix Model::Model::jacobian(StateVector &x)
{
  // initialize the Jacobian as a zero matrix.
  StateMatrix J = StateMatrix::Zero(x.rows(), x.cols());

  // Loop through every right hand side contribution added to the model and keep adding to the Jacobian.
  for (auto & rhs_contribution : rhs_contributions)
  {
    rhs_contribution->add_contribution_to_jacobian(x, J);
  }

  return J;
}