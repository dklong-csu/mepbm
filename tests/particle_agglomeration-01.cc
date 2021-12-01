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
using Vector1 = Eigen::Matrix<Real1, Eigen::Dynamic, 1>;
using Vector2 = Eigen::Matrix<Real2, Eigen::Dynamic, 1>;



template<typename InputType>
void
check_vars(InputType & rxn)
{
  std::cout << std::boolalpha << (rxn.particleA.index_start == 2) << std::endl;
  std::cout << std::boolalpha << (rxn.particleA.index_end == 3) << std::endl;
  std::cout << std::boolalpha << (rxn.particleA.first_size == 1) << std::endl;

  std::cout << std::boolalpha << (rxn.particleB.index_start == 2) << std::endl;
  std::cout << std::boolalpha << (rxn.particleB.index_end == 3) << std::endl;
  std::cout << std::boolalpha << (rxn.particleB.first_size == 1) << std::endl;

  std::cout << std::boolalpha << (rxn.reaction_rate == 1.5) << std::endl;

  std::cout << std::boolalpha << (rxn.max_particle_size == 6) << std::endl;

  std::cout << std::boolalpha << (rxn.growth_kernel(10) == 25.) << std::endl;

  std::cout << std::boolalpha << (rxn.reactants.size() == 0) << std::endl;
  std::cout << std::boolalpha << (rxn.products.size() == 0) << std::endl;
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
  Model::Particle B(2, 3, 1);

  const std::vector<ReactionPair> reactants;
  const std::vector<ReactionPair> products;

  const unsigned int max_particle_size = 6;

  // Do a test for each of the intended types for matrices and floating point number combination
  {
    const Real1 reaction_rate = 1.5;
    Model::ParticleAgglomeration<Real1, SparseMatrix1> rxn(B,
                                                           B,
                                                           reaction_rate,
                                                           max_particle_size,
                                                           &growth_kernel<Real1>,
                                                           reactants,
                                                           products);
    check_vars(rxn);
    std::cout << std::endl;
  }

  {
    const Real1 reaction_rate = 1.5;
    Model::ParticleAgglomeration<Real1, DenseMatrix1> rxn(B,
                                                   B,
                                                   reaction_rate,
                                                   max_particle_size,
                                                   &growth_kernel<Real1>,
                                                   reactants,
                                                   products);

    check_vars(rxn);
    std::cout << std::endl;
  }

  {
    const Real2 reaction_rate = 1.5;
    Model::ParticleAgglomeration<Real2, SparseMatrix2> rxn(B,
                                                    B,
                                                    reaction_rate,
                                                    max_particle_size,
                                                    &growth_kernel<Real2>,
                                                    reactants,
                                                    products);

    check_vars(rxn);
    std::cout << std::endl;
  }

  {
    const Real2 reaction_rate = 1.5;
    Model::ParticleAgglomeration<Real2, DenseMatrix2> rxn(B,
                                                   B,
                                                   reaction_rate,
                                                   max_particle_size,
                                                   &growth_kernel<Real2>,
                                                   reactants,
                                                   products);

    check_vars(rxn);
  }
}