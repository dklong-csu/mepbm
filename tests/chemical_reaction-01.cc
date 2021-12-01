#include "chemical_reaction.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <utility>
#include <vector>

/*
 * This tests all of the member variables in the ChemicalReaction class
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
  auto reactants = rxn.reactants;
  // Should have exactly two reactants
  std::cout << std::boolalpha << (reactants.size() == 2) << std::endl;

  auto products = rxn.products;
  // Should have exactly one product
  std::cout << std::boolalpha << (products.size() == 1) << std::endl;

  auto rxn_rate = rxn.reaction_rate;
  // reaction rate should be exactly 1.5
  std::cout << std::boolalpha << (rxn_rate == 1.5) << std::endl;

  // First entry in reactants should correspond to 1A, which has vector index 0.
  std::cout << std::boolalpha << (reactants[0].first.index == 0) << std::endl;
  std::cout << std::boolalpha << (reactants[0].second == 1) << std::endl;

  // Second entry in reactants should correspond to 2B, which has vector index 1.
  std::cout << std::boolalpha << (reactants[1].first.index == 1) << std::endl;
  std::cout << std::boolalpha << (reactants[1].second == 2) << std::endl;

  // First (and only) entry in products should correspond to 3C, which has vector index 2.
  std::cout << std::boolalpha << (products[0].first.index == 2) << std::endl;
  std::cout << std::boolalpha << (products[0].second == 3) << std::endl;

}



int main ()
{
  /*
   * The chemical reaction
   *    A + 2B ->[k=1.5] 3C
   * is used for this example.
   */
  Model::Species A(0);
  Model::Species B(1);
  Model::Species C(2);

  ReactionPair rxnA = {A,1};
  ReactionPair rxnB = {B,2};
  ReactionPair rxnC = {C,3};

  double rxn_rate = 1.5;

  std::vector<ReactionPair> reactants = {rxnA, rxnB};
  std::vector<ReactionPair> products = {rxnC};

  // Do a test for each of the intended types for matrices and floating point number combination
  {
    Model::ChemicalReaction<Real1, SparseMatrix1> rxn(reactants, products, rxn_rate);
    check_rxn_members(rxn);
  }

  {
    Model::ChemicalReaction<Real1, DenseMatrix1> rxn(reactants, products, rxn_rate);
    check_rxn_members(rxn);
  }

  {
    Model::ChemicalReaction<Real2, SparseMatrix2> rxn(reactants, products, rxn_rate);
    check_rxn_members(rxn);
  }

  {
    Model::ChemicalReaction<Real2, DenseMatrix2> rxn(reactants, products, rxn_rate);
    check_rxn_members(rxn);
  }
}