#include "src/chemical_reaction.h"
#include "src/particle_growth.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <utility>
#include <vector>

/*
 * This tests all of the member variables in the ParticleGrowth class
 */

using Real = realtype;
using SparseMatrix = Eigen::SparseMatrix<Real>;
using DenseMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using ReactionPair = std::pair<MEPBM::Species, unsigned int>;



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
  MEPBM::Species A(0);
  MEPBM::Species L(1);
  MEPBM::Particle B(2, 5, 3);

  ReactionPair rxnA = {A,2};
  ReactionPair rxnL = {L, 3};
  const double reaction_rate = 1.5;

  const std::vector<ReactionPair> reactants = {rxnA};
  const std::vector<ReactionPair> products = {rxnL};

  const unsigned int growth_amount = 1;
  const unsigned int max_particle_size = 6;


  // Test for Dense and Sparse matrices
  MEPBM::ParticleGrowth<Real, DenseMatrix> growth_dense(B,
                                                        reaction_rate,
                                                        growth_amount,
                                                        max_particle_size,
                                                        &growth_kernel<Real>,
                                                            reactants,
                                                            products);

  check_rxn_members(growth_dense);


  MEPBM::ParticleGrowth<Real, SparseMatrix > growth_sparse(B,
                                                        reaction_rate,
                                                        growth_amount,
                                                        max_particle_size,
                                                        &growth_kernel<Real>,
                                                        reactants,
                                                        products);

  check_rxn_members(growth_sparse);

}