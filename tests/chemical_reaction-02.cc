#include "chemical_reaction.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <utility>
#include <vector>

/*
 * This tests all of the member functions in the ChemicalReaction class
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



template<typename InputType, typename VectorType, typename MatrixType>
void
check_rxn_fcns(InputType & rxn)
{
  // Check the rhs function
  VectorType x(3);
  x << 1, 2, 0;
  /*
   * ODEs should be
   * dA/dt = -kA*B^2  = -6
   * dB/dt = -2kA*B^2 = -12
   * dC/dt = 3kA*B^2  = 18
   */
  VectorType rhs(3);
  rhs.setZero();
  rxn.add_contribution_to_rhs(x,rhs);
  std::cout << rhs(0) << std::endl;
  std::cout << rhs(1) << std::endl;
  std::cout << rhs(2) << std::endl;

  // Check the Jacobian function
  MatrixType J(3,3);
  J.setZero();
  /*
   * Jacobian should be
   *    |dA'/dA  dA'/dB  dA'/dC|    | -kB^2    -2kA*B   0|    | -6   -6   0|
   *    |dB'/dA  dB'/dB  dB'/dC|  = | -k2B^2   -4kA*B   0|  = | -12  -12  0|
   *    |dC'/dA  dC'/dB  dC'/dC|    | 3kB^2    6kA*B    0|    | 18   18   0|
   */
  rxn.add_contribution_to_jacobian(x,J);
  std::cout << J.coeffRef(0,0) << std::endl;
  std::cout << J.coeffRef(0,1) << std::endl;
  std::cout << J.coeffRef(0,2) << std::endl;

  std::cout << J.coeffRef(1,0) << std::endl;
  std::cout << J.coeffRef(1,1) << std::endl;
  std::cout << J.coeffRef(1,2) << std::endl;

  std::cout << J.coeffRef(2,0) << std::endl;
  std::cout << J.coeffRef(2,1) << std::endl;
  std::cout << J.coeffRef(2,2) << std::endl;
}



template<typename InputType, typename Real>
void
check_sparse_rxn_fcns(InputType & rxn)
{
  std::vector<Eigen::Triplet<Real>> triplet_list;
  rxn.add_nonzero_to_jacobian(triplet_list);
  for (auto t : triplet_list)
  {
    std::cout << t.row() << ", " << t.col() << std::endl;
  }

  unsigned int n_nonzero = 0;
  rxn.update_num_nonzero(n_nonzero);
  std::cout << n_nonzero << std::endl;
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

  const double rxn_rate = 1.5;

  const std::vector<ReactionPair> reactants = {rxnA, rxnB};
  const std::vector<ReactionPair> products = {rxnC};

  // Do a test for each of the intended types for matrices and floating point number combination

  // Sparse double
  Model::ChemicalReaction<Real1, SparseMatrix1> rxn_s1(reactants, products, rxn_rate);
  check_rxn_fcns<Model::ChemicalReaction<Real1, SparseMatrix1>, Vector1, SparseMatrix1>(rxn_s1);
  // Sparse matrices make sense to check the other two functions
  check_sparse_rxn_fcns<Model::ChemicalReaction<Real1, SparseMatrix1>, Real1>(rxn_s1);
  std::cout << std::endl;



  // Dense double
  Model::ChemicalReaction<Real1, DenseMatrix1> rxn_d1(reactants, products, rxn_rate);
  check_rxn_fcns<Model::ChemicalReaction<Real1, DenseMatrix1>, Vector1, DenseMatrix1>(rxn_d1);
  std::cout << std::endl;



  // Sparse float
  Model::ChemicalReaction<Real2, SparseMatrix2> rxn_s2(reactants, products, rxn_rate);
  check_rxn_fcns<Model::ChemicalReaction<Real2, SparseMatrix2>, Vector2, SparseMatrix2>(rxn_s2);
  // Sparse matrices make sense to check the other two functions
  check_sparse_rxn_fcns<Model::ChemicalReaction<Real2, SparseMatrix2>, Real2>(rxn_s2);
  std::cout << std::endl;



  // Dense float
  Model::ChemicalReaction<Real2, DenseMatrix2> rxn_d2(reactants, products, rxn_rate);
  check_rxn_fcns<Model::ChemicalReaction<Real2, DenseMatrix2>, Vector2, DenseMatrix2>(rxn_d2);

}