#include "src/chemical_reaction.h"
#include "src/particle.h"
#include "src/species.h"
#include "src/particle_agglomeration.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <utility>
#include <vector>

/*
 * This tests all of the member variables in the ParticleAgglomeration class
 */

using Real = realtype;
using SparseMatrix = Eigen::SparseMatrix<Real>;
using DenseMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using ReactionPair = std::pair<MEPBM::Species, unsigned int>;



template<typename InputType>
void
check_vars(InputType & rxn)
{
  std::cout << std::boolalpha << (rxn.particleA.index_start == 2) << std::endl;
  std::cout << std::boolalpha << (rxn.particleA.index_end == 3) << std::endl;
  std::cout << std::boolalpha << (rxn.particleA.first_size == 1) << std::endl;

  std::cout << std::boolalpha << (rxn.particleB.index_start == 4) << std::endl;
  std::cout << std::boolalpha << (rxn.particleB.index_end == 7) << std::endl;
  std::cout << std::boolalpha << (rxn.particleB.first_size == 3) << std::endl;

  std::cout << std::boolalpha << (rxn.reaction_rate == 1.5) << std::endl;

  std::cout << std::boolalpha << (rxn.max_particle_size == 6) << std::endl;

  std::cout << std::boolalpha << (rxn.growth_kernel(10) == 25.) << std::endl;

  std::cout << std::boolalpha << (rxn.reactants.size() == 1) << std::endl;
  std::cout << std::boolalpha << (rxn.reactants[0].first.index == 0) << std::endl;
  std::cout << std::boolalpha << (rxn.reactants[0].second == 2) << std::endl;

  std::cout << std::boolalpha << (rxn.products.size() == 1) << std::endl;
  std::cout << std::boolalpha << (rxn.products[0].first.index == 1) << std::endl;
  std::cout << std::boolalpha << (rxn.products[0].second == 1) << std::endl;
}



Real
growth_kernel(const unsigned int size)
{
  return size * 2.5;
}



int main ()
{
  MEPBM::Particle B(2, 3, 1);
  MEPBM::Particle C(4,7,3);

  MEPBM::Species X(0);
  MEPBM::Species Y(1);
  const std::vector<ReactionPair> reactants = {{X,2}};
  const std::vector<ReactionPair> products = {{Y,1}};

  const unsigned int max_particle_size = 6;

  const Real reaction_rate = 1.5;

  // Test
  MEPBM::ParticleAgglomeration<Real, DenseMatrix> agglom_dense(B,C,reaction_rate,max_particle_size,&growth_kernel,reactants, products);
  check_vars(agglom_dense);

  MEPBM::ParticleAgglomeration<Real, SparseMatrix> agglom_sparse(B,C,reaction_rate,max_particle_size,&growth_kernel,reactants, products);
  check_vars(agglom_sparse);
}