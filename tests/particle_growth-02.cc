#include "src/chemical_reaction.h"
#include "src/particle_growth.h"
#include "src/create_nvector.h"
#include "src/create_sunmatrix.h"
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
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;



template<typename InputType>
void
check_rhs(InputType & rxn)
{
  auto x = MEPBM::create_eigen_nvector<Vector>(6);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 2,3,4,5,6,7;

  auto x_dot = MEPBM::create_eigen_nvector<Vector>(6);
  x_dot->ops->nvconst(0., x_dot);

  /*
   * ODE should be
   *    dA/dt = -2A^2*k*sum(r(i)*B_i)= -12*(30+50+75+105)       = -3120
   *    dL/dt = 3A^2*k*sum(r(i)*B_i) = 18*(30+50+75+105)        = 4680
   *    dB_3/dt = -r(3)*k*A^2*B_3                               = -180
   *    dB_4/dt = r(3)*k*A^2*B_3 - r(4)*k*A^2*B_4   = 180 - 300 = -120
   *    dB_5/dt = r(4)*k*A^2*B_4 - r(5)*k*A^2*B_5   = 300 - 450 = -150
   *    dB_6/dt = r(5)*k*A^2*B_5 - r(6)*k*A^2*B_6   = 450 - 630 = -180
   */
  auto rhs = rxn.rhs_function();
  auto err = rhs(0.0, x, x_dot, nullptr);

  // Should not get an error, so check err = 0
  std::cout << err << std::endl;

  // Output resulting rhs vector
  auto rhs_vec = *static_cast<Vector*>(x_dot->content);
  std::cout << rhs_vec << std::endl;

}



template<typename InputType, typename MatrixType>
void
check_jacobian(InputType & rxn)
{
  auto x = MEPBM::create_eigen_nvector<Vector>(6);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 2,3,4,5,6,7;

  auto x_dot = MEPBM::create_eigen_nvector<Vector>(6);
  x_dot->ops->nvconst(0., x_dot);

  auto J = MEPBM::create_eigen_sunmatrix<MatrixType>(6,6);
  J->ops->zero(J);
  /*
   * Jacobian should be
   *    |dA'/dA    dA'/dL    dA'/dB_3    dA'/dB_4    dA'/dB_5    dA'/dB_6  |   |-3120  0  -90  -120  -150 -180|
   *    |dL'/dA    dL'/dL    dL'/dB_3    dL'/dB_4    dL'/dB_5    dL'/dB_6  |   |4680   0  135   180   225  270|
   *    |dB_3'/dA  dB_3'/dL  dB_3'/dB_3  dB_3'/dB_4  dB_3'/dB_5  dB_3'/dB_6| = |-180   0  -45   0     0    0  |
   *    |dB_4'/dA  dB_4'/dL  dB_4'/dB_3  dB_4'/dB_4  dB_4'/dB_5  dB_4'/dB_6|   |-120   0  45    -60   0    0  |
   *    |dB_5'/dA  dB_5'/dL  dB_5'/dB_3  dB_5'/dB_4  dB_5'/dB_5  dB_5'/dB_6|   |-150   0  0      60   -75  0  |
   *    |dB_6'/dA  dB_6'/dL  dB_6'/dB_3  dB_6'/dB_4  dB_6'/dB_5  dB_6'/dB_6|   |-180   0  0      0     75  -90|
   */
  auto J_fcn = rxn.jacobian_function();
  auto tmp1 = MEPBM::create_eigen_nvector<Vector>(6);
  auto tmp2 = MEPBM::create_eigen_nvector<Vector>(6);
  auto tmp3 = MEPBM::create_eigen_nvector<Vector>(6);
  auto err = J_fcn(0.0, x, x_dot, J, nullptr, tmp1, tmp2, tmp3);

  // Should not have an error so check for err=0
  std::cout << err << std::endl;

  // Check the Jacobian
  MatrixType J_mat = *static_cast<MatrixType*>(J->content);
  // Output manually because outputting a Sparse Matrix using << gives undesired information
  const auto r = J_mat.rows();
  const auto c = J_mat.cols();
  for (unsigned int i=0; i<r; ++i)
  {
    for (unsigned int j=0; j<c; ++j)
    {
      std::cout << J_mat.coeffRef(i,j) << ' ';
    }
    std::cout << std::endl;
  }
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

  check_rhs(growth_dense);
  check_jacobian<MEPBM::ParticleGrowth<Real, DenseMatrix>, DenseMatrix>(growth_dense);


  MEPBM::ParticleGrowth<Real, SparseMatrix> growth_sparse(B,
                                                        reaction_rate,
                                                        growth_amount,
                                                        max_particle_size,
                                                        &growth_kernel<Real>,
                                                        reactants,
                                                        products);

  check_rhs(growth_sparse);
  check_jacobian<MEPBM::ParticleGrowth<Real, SparseMatrix>, SparseMatrix>(growth_sparse);
}