#include "src/chemical_reaction.h"
#include "src/species.h"
#include "src/create_nvector.h"
#include "src/create_sunmatrix.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <utility>
#include <vector>

/*
 * This tests all of the member functions in the ChemicalReaction class
 */

using Real = realtype;
using SparseMatrix1 = Eigen::SparseMatrix<Real>;
using SparseMatrix2 = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using DenseMatrix1 = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using ReactionPair = std::pair<MEPBM::Species, unsigned int>;
using Vector1 = Eigen::Matrix<Real, Eigen::Dynamic, 1>;



template<typename InputType, typename VectorType, typename MatrixType>
void
check_dense(InputType & rxn)
{
  // Create necessary objects to pass to the functions
  // Check the rhs function
  auto x = MEPBM::create_eigen_nvector<VectorType>(3);
  auto x_vec = static_cast<VectorType*>(x->content);
  *x_vec << 1, 2, 0;
  /*
   * ODEs should be
   * dA/dt = -kA*B^2  = -6
   * dB/dt = -2kA*B^2 = -12
   * dC/dt = 3kA*B^2  = 18
   */
  auto rhs = MEPBM::create_eigen_nvector<VectorType>(3);
  auto rhs_vec = static_cast<VectorType*>(rhs->content);
  *rhs_vec << 0.,0.,0.;

  auto rhs_fcn = rxn.rhs_function();
  auto err_rhs = rhs_fcn(0.0, x, rhs, nullptr);

  std::cout << (*rhs_vec)(0) << std::endl;
  std::cout << (*rhs_vec)(1) << std::endl;
  std::cout << (*rhs_vec)(2) << std::endl;

  // Check the Jacobian function
  auto J = MEPBM::create_eigen_sunmatrix<MatrixType>(3,3);
  J->ops->zero(J);
  /*
   * Jacobian should be
   *    |dA'/dA  dA'/dB  dA'/dC|    | -kB^2    -2kA*B   0|    | -6   -6   0|
   *    |dB'/dA  dB'/dB  dB'/dC|  = | -k2B^2   -4kA*B   0|  = | -12  -12  0|
   *    |dC'/dA  dC'/dB  dC'/dC|    | 3kB^2    6kA*B    0|    | 18   18   0|
   */
  auto jac_fcn = rxn.jacobian_function();
  auto tmp1 = MEPBM::create_eigen_nvector<VectorType>(3);
  auto tmp2 = MEPBM::create_eigen_nvector<VectorType>(3);
  auto tmp3 = MEPBM::create_eigen_nvector<VectorType>(3);
  auto err_j = jac_fcn(0.0, x, rhs, J, nullptr, tmp1, tmp2, tmp3);

  auto J_mat = *static_cast<MatrixType*>(J->content);

  std::cout << J_mat << std::endl;

  x->ops->nvdestroy(x);
  rhs->ops->nvdestroy(rhs);
  J->ops->destroy(J);
  tmp1->ops->nvdestroy(tmp1);
  tmp2->ops->nvdestroy(tmp2);
  tmp3->ops->nvdestroy(tmp3);
}



template<typename InputType, typename VectorType, typename MatrixType>
void
check_sparse(InputType & rxn)
{
  // Create necessary objects to pass to the functions
  // Check the rhs function
  auto x = MEPBM::create_eigen_nvector<VectorType>(3);
  auto x_vec = static_cast<VectorType*>(x->content);
  *x_vec << 1, 2, 0;
  /*
   * ODEs should be
   * dA/dt = -kA*B^2  = -6
   * dB/dt = -2kA*B^2 = -12
   * dC/dt = 3kA*B^2  = 18
   */
  auto rhs = MEPBM::create_eigen_nvector<VectorType>(3);
  auto rhs_vec = static_cast<VectorType*>(rhs->content);
  *rhs_vec << 0.,0.,0.;

  auto rhs_fcn = rxn.rhs_function();
  auto err_rhs = rhs_fcn(0.0, x, rhs, nullptr);

  std::cout << (*rhs_vec)(0) << std::endl;
  std::cout << (*rhs_vec)(1) << std::endl;
  std::cout << (*rhs_vec)(2) << std::endl;

  // Check the Jacobian function
  auto J = MEPBM::create_eigen_sunmatrix<MatrixType>(3,3);
  J->ops->zero(J);
  /*
   * Jacobian should be
   *    |dA'/dA  dA'/dB  dA'/dC|    | -kB^2    -2kA*B   0|    | -6   -6   0|
   *    |dB'/dA  dB'/dB  dB'/dC|  = | -k2B^2   -4kA*B   0|  = | -12  -12  0|
   *    |dC'/dA  dC'/dB  dC'/dC|    | 3kB^2    6kA*B    0|    | 18   18   0|
   */
  auto jac_fcn = rxn.jacobian_function();
  std::vector<Eigen::Triplet<Real>> triplet_list;
  auto err_j = jac_fcn(x, triplet_list, J);
  auto J_mat = *static_cast<MatrixType*>(J->content);
  J_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());

  std::cout << J_mat << std::endl;


  x->ops->nvdestroy(x);
  rhs->ops->nvdestroy(rhs);
  J->ops->destroy(J);
}



int main ()
{
  /*
   * The chemical reaction
   *    A + 2B ->[k=1.5] 3C
   * is used for this example.
   */
  MEPBM::Species A(0);
  MEPBM::Species B(1);
  MEPBM::Species C(2);

  ReactionPair rxnA = {A,1};
  ReactionPair rxnB = {B,2};
  ReactionPair rxnC = {C,3};

  const double rxn_rate = 1.5;

  const std::vector<ReactionPair> reactants = {rxnA, rxnB};
  const std::vector<ReactionPair> products = {rxnC};

  // Do a test for each of the intended types for matrices and floating point number combination

  // Sparse
  MEPBM::ChemicalReaction<Real, SparseMatrix1> rxn_s1(reactants, products, rxn_rate);
  check_sparse<MEPBM::ChemicalReaction<Real, SparseMatrix1>, Vector1, SparseMatrix1>(rxn_s1);


  // Sparse
  MEPBM::ChemicalReaction<Real, SparseMatrix2> rxn_s2(reactants, products, rxn_rate);
  check_sparse<MEPBM::ChemicalReaction<Real, SparseMatrix2>, Vector1, SparseMatrix2>(rxn_s2);


  // Dense
  MEPBM::ChemicalReaction<Real, DenseMatrix1> rxn_d1(reactants, products, rxn_rate);
  check_dense<MEPBM::ChemicalReaction<Real, DenseMatrix1>, Vector1, DenseMatrix1>(rxn_d1);
}