#include "src/chemical_reaction.h"
#include "src/particle.h"
#include "src/species.h"
#include "src/particle_agglomeration.h"
#include "src/create_nvector.h"
#include "src/create_sunmatrix.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <utility>
#include <vector>
#include <iomanip>

/*
 * This tests all of the member functions in the ParticleAgglomeration class
 */

using Real = realtype;
using SparseMatrix = Eigen::SparseMatrix<Real>;
using SparseMatrix2 = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using DenseMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using ReactionPair = std::pair<MEPBM::Species, unsigned int>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;



template<typename InputType>
void
check_rhs(InputType & rxn)
{
  auto x = MEPBM::create_eigen_nvector<Vector>(6);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 1,2,3,4,5,6;

  auto x_dot = MEPBM::create_eigen_nvector<Vector>(6);
  x_dot->ops->nvconst(0., x_dot);

  auto rhs = rxn.rhs_function();
  auto err = rhs(0.0, x, x_dot, nullptr);

  // Should not get an error, so check err = 0
  std::cout << err << std::endl;

  // Output resulting rhs vector
  auto rhs_vec = *static_cast<Vector*>(x_dot->content);
  std::cout << std::setprecision(10) << rhs_vec << std::endl;

  x->ops->nvdestroy(x);
  x_dot->ops->nvdestroy(x_dot);
}



template<typename InputType, typename MatrixType>
void
check_dense(InputType & rxn)
{
  auto x = MEPBM::create_eigen_nvector<Vector>(6);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 1,2,3,4,5,6;

  auto x_dot = MEPBM::create_eigen_nvector<Vector>(6);
  x_dot->ops->nvconst(0., x_dot);

  auto J = MEPBM::create_eigen_sunmatrix<MatrixType>(6,6);
  J->ops->zero(J);

  auto J_fcn = rxn.jacobian_function();
  auto tmp1 = MEPBM::create_eigen_nvector<Vector>(6);
  auto tmp2 = MEPBM::create_eigen_nvector<Vector>(6);
  auto tmp3 = MEPBM::create_eigen_nvector<Vector>(6);
  auto err = J_fcn(0.0, x, x_dot, J, nullptr, tmp1, tmp2, tmp3);

  // Should not have an error so check for err=0
  std::cout << err << std::endl;

  // Check the Jacobian
  auto J_mat = *static_cast<MatrixType*>(J->content);
  std::cout << J_mat << std::endl;

  x->ops->nvdestroy(x);
  x_dot->ops->nvdestroy(x_dot);
  J->ops->destroy(J);
  tmp1->ops->nvdestroy(tmp1);
  tmp2->ops->nvdestroy(tmp2);
  tmp3->ops->nvdestroy(tmp3);
}



template<typename InputType, typename MatrixType>
void
check_sparse(InputType & rxn)
{
  auto x = MEPBM::create_eigen_nvector<Vector>(6);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 1,2,3,4,5,6;

  auto x_dot = MEPBM::create_eigen_nvector<Vector>(6);
  x_dot->ops->nvconst(0., x_dot);

  auto J = MEPBM::create_eigen_sunmatrix<MatrixType>(6,6);
  J->ops->zero(J);

  auto J_fcn = rxn.jacobian_function();
  std::vector<Eigen::Triplet<Real>> triplet_list;
  auto err = J_fcn(x, triplet_list, J);

  // Should not have an error so check for err=0
  std::cout << err << std::endl;

  // Check the Jacobian
  auto J_mat = *static_cast<MatrixType*>(J->content);
  J_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
  std::cout << J_mat << std::endl;

  x->ops->nvdestroy(x);
  x_dot->ops->nvdestroy(x_dot);
  J->ops->destroy(J);
}



Real
growth_kernel(const unsigned int sizeA, const unsigned int sizeB)
{
  return 1.5 * sizeA * 2.5 * sizeB * 2.5;
}



int main ()
{
  MEPBM::Particle B(2, 3, 1);
  MEPBM::Particle C(4,5,3);

  MEPBM::Species X(0);
  MEPBM::Species Y(1);
  const std::vector<ReactionPair> reactants = {{X,2}};
  const std::vector<ReactionPair> products = {{Y,1}};

  const unsigned int max_particle_size = 4;

  // Test
  MEPBM::ParticleAgglomeration<Real, DenseMatrix> agglom_dense(B,C,max_particle_size,&growth_kernel,reactants, products);
  check_rhs(agglom_dense);
  check_dense<MEPBM::ParticleAgglomeration<Real, DenseMatrix>, DenseMatrix>(agglom_dense);


  MEPBM::ParticleAgglomeration<Real, SparseMatrix> agglom_sparse(B,C,max_particle_size,&growth_kernel,reactants, products);
  check_rhs(agglom_sparse);
  check_sparse<MEPBM::ParticleAgglomeration<Real, SparseMatrix >, SparseMatrix>(agglom_sparse);


  MEPBM::ParticleAgglomeration<Real, SparseMatrix2> agglom_sparse2(B,C,max_particle_size,&growth_kernel,reactants, products);
  check_rhs(agglom_sparse2);
  check_sparse<MEPBM::ParticleAgglomeration<Real, SparseMatrix2 >, SparseMatrix2>(agglom_sparse2);

}