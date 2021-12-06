#include "chemical_reaction.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <utility>
#include <vector>

/*
 * This tests all of the member variables in the ParticleGrowth class
 */

using Real1 = double;
using Real2 = float;
using SparseMatrix1 = Eigen::SparseMatrix<Real1>;
using SparseMatrix2 = Eigen::SparseMatrix<Real2>;
using DenseMatrix1 = Eigen::Matrix<Real1, Eigen::Dynamic, Eigen::Dynamic>;
using DenseMatrix2 = Eigen::Matrix<Real2, Eigen::Dynamic, Eigen::Dynamic>;
using ReactionPair = std::pair<Model::Species, unsigned int>;



template<typename InputType>
void
check_rxn_members(const InputType & rxn)
{
  std::cout << std::boolalpha << (rxn.particle.index_start == 2) << std::endl;
  std::cout << std::boolalpha << (rxn.particle.index_end == 5) << std::endl;
  std::cout << std::boolalpha << (rxn.particle.first_size == 3) << std::endl;

  std::cout << std::boolalpha << (rxn.reaction_rate == 1.5) << std::endl;

  std::cout << std::boolalpha << (rxn.growth_amount == 1) << std::endl;

  std::cout << std::boolalpha << (rxn.max_particle_size == 6) << std::endl;

  std::cout << std::boolalpha << (rxn.growth_kernel(2) == 5.) << std::endl;

  std::cout << std::boolalpha << (rxn.reactants.size() == 1) << std::endl;
  std::cout << std::boolalpha << (rxn.reactants[0].first.index == 0) << std::endl;
  std::cout << std::boolalpha << (rxn.reactants[0].second == 2) << std::endl;

  std::cout << std::boolalpha << (rxn.products.size() == 1) << std::endl;
  std::cout << std::boolalpha << (rxn.products[0].first.index == 1) << std::endl;
  std::cout << std::boolalpha << (rxn.products[0].second == 3) << std::endl;
}


template <typename Real>
Real
growth_kernel(const unsigned int size)
{
  return size * 2.5;
}



int main ()
{
  /*
   * The chemical reaction
   *    2A + B ->[k=1.5*r(i)] B + 3L
   * is used for this example.
   */
  Model::Species A(0);
  Model::Species L(1);
  Model::Particle B(2, 5, 3);

  ReactionPair rxnA = {A,2};
  ReactionPair rxnL = {L, 3};
  const double reaction_rate = 1.5;

  const std::vector<ReactionPair> reactants = {rxnA};
  const std::vector<ReactionPair> products = {rxnL};

  const unsigned int growth_amount = 1;
  const unsigned int max_particle_size = 6;


  // Do a test for each of the intended types for matrices and floating point number combination
  {
    Model::ParticleGrowth<Real1, SparseMatrix1> rxn(B,
                                                    reaction_rate,
                                                    growth_amount,
                                                    max_particle_size,
                                                    &growth_kernel<Real1>,
                                                        reactants,
                                                        products);
    check_rxn_members(rxn);
    std::cout << std::endl;
  }

  {
    Model::ParticleGrowth<Real1, DenseMatrix1> rxn(B,
                                                   reaction_rate,
                                                   growth_amount,
                                                   max_particle_size,
                                                   &growth_kernel<Real1>,
                                                   reactants,
                                                   products);
    check_rxn_members(rxn);
    std::cout << std::endl;
  }

  {
    Model::ParticleGrowth<Real2, SparseMatrix2> rxn(B,
                                                    reaction_rate,
                                                    growth_amount,
                                                    max_particle_size,
                                                    &growth_kernel<Real1>,
                                                    reactants,
                                                    products);
    check_rxn_members(rxn);
    std::cout << std::endl;
  }

  {
    Model::ParticleGrowth<Real2, DenseMatrix2> rxn(B,
                                                   reaction_rate,
                                                   growth_amount,
                                                   max_particle_size,
                                                   &growth_kernel<Real1>,
                                                   reactants,
                                                   products);
    check_rxn_members(rxn);
  }
}