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



template<typename InputType, typename VectorType>
void
check_rhs(InputType & rxn)
{
  VectorType x(6);
  x << 2,3,4,5,6,7;
  VectorType rhs(6);
  rhs.setZero();
  /*
   * ODE should be
   *    dA/dt = -2A^2*k*sum(r(i)*B_i)= -12*(30+50+75+105)       = -3120
   *    dL/dt = 3A^2*k*sum(r(i)*B_i) = 18*(30+50+75+105)        = 4680
   *    dB_3/dt = -r(3)*k*A^2*B_3                               = -180
   *    dB_4/dt = r(3)*k*A^2*B_3 - r(4)*k*A^2*B_4   = 180 - 300 = -120
   *    dB_5/dt = r(4)*k*A^2*B_4 - r(5)*k*A^2*B_5   = 300 - 450 = -150
   *    dB_6/dt = r(5)*k*A^2*B_5 - r(6)*k*A^2*B_6   = 450 - 630 = -180
   */
  rxn.add_contribution_to_rhs(x,rhs);
  for (unsigned int i=0; i<rhs.size(); ++i)
  {
    std::cout << rhs(i) << std::endl;
  }

}



template<typename InputType, typename VectorType, typename MatrixType>
void
check_jacobian(InputType & rxn)
{
  VectorType x(6);
  x << 2,3,4,5,6,7;
  MatrixType J(6,6);
  J.setZero();
  /*
   * Jacobian should be
   *    |dA'/dA    dA'/dL    dA'/dB_3    dA'/dB_4    dA'/dB_5    dA'/dB_6  |   |-3120  0  -90  -120  -150 -180|
   *    |dL'/dA    dL'/dL    dL'/dB_3    dL'/dB_4    dL'/dB_5    dL'/dB_6  |   |4680   0  135   180   225  270|
   *    |dB_3'/dA  dB_3'/dL  dB_3'/dB_3  dB_3'/dB_4  dB_3'/dB_5  dB_3'/dB_6| = |-180   0  -45   0     0    0  |
   *    |dB_4'/dA  dB_4'/dL  dB_4'/dB_3  dB_4'/dB_4  dB_4'/dB_5  dB_4'/dB_6|   |-120   0  45    -60   0    0  |
   *    |dB_5'/dA  dB_5'/dL  dB_5'/dB_3  dB_5'/dB_4  dB_5'/dB_5  dB_5'/dB_6|   |-150   0  0      60   -75  0  |
   *    |dB_6'/dA  dB_6'/dL  dB_6'/dB_3  dB_6'/dB_4  dB_6'/dB_5  dB_6'/dB_6|   |-180   0  0      0     75  -90|
   */
  rxn.add_contribution_to_jacobian(x,J);
  for (unsigned int i=0; i<J.rows(); ++i)
  {
    for (unsigned int j=0; j<J.cols(); ++j)
    {
      std::cout << J.coeffRef(i,j) << std::endl;
    }
  }
}



template<typename InputType, typename Real>
void
check_add_nonzero(InputType & rxn)
{
  std::vector<Eigen::Triplet<Real>> triplet_list;
  rxn.add_nonzero_to_jacobian(triplet_list);
  for (auto t : triplet_list)
  {
    std::cout << t.row() << ", " << t.col() << std::endl;
  }
}



template<typename InputType>
void
check_update_nonzero(InputType & rxn)
{
  unsigned int n_nonzero = 0;
  rxn.update_num_nonzero(n_nonzero);
  // 21 is the exact number of nonzeros but function is designed to overestimate for simplicity
  std::cout << std::boolalpha << (n_nonzero >= 21) << std::endl;
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

  const std::vector<ReactionPair> reactants = {rxnA};
  const std::vector<ReactionPair> products = {rxnL};

  const unsigned int growth_amount = 1;
  const unsigned int max_particle_size = 6;


  // Do a test for each of the intended types for matrices and floating point number combination
  {
    const Real1 reaction_rate = 1.5;
    Model::ParticleGrowth<Real1, SparseMatrix1> rxn(B,
                                                    reaction_rate,
                                                    growth_amount,
                                                    max_particle_size,
                                                    &growth_kernel<Real1>,
                                                    reactants,
                                                    products);
    check_rhs<Model::ParticleGrowth<Real1, SparseMatrix1>, Vector1>(rxn);
    check_jacobian<Model::ParticleGrowth<Real1, SparseMatrix1>, Vector1, SparseMatrix1>(rxn);
    check_add_nonzero<Model::ParticleGrowth<Real1, SparseMatrix1>, Real1>(rxn);
    check_update_nonzero(rxn);
    std::cout << std::endl;
  }

  {
    const Real1 reaction_rate = 1.5;
    Model::ParticleGrowth<Real1, DenseMatrix1> rxn(B,
                                                   reaction_rate,
                                                   growth_amount,
                                                   max_particle_size,
                                                   &growth_kernel<Real1>,
                                                   reactants,
                                                   products);

    check_rhs<Model::ParticleGrowth<Real1, DenseMatrix1>, Vector1>(rxn);
    check_jacobian<Model::ParticleGrowth<Real1, DenseMatrix1>, Vector1, DenseMatrix1>(rxn);
    std::cout << std::endl;
  }

  {
    const Real2 reaction_rate = 1.5;
    Model::ParticleGrowth<Real2, SparseMatrix2> rxn(B,
                                                    reaction_rate,
                                                    growth_amount,
                                                    max_particle_size,
                                                    &growth_kernel<Real2>,
                                                    reactants,
                                                    products);

    check_rhs<Model::ParticleGrowth<Real2, SparseMatrix2>, Vector2>(rxn);
    check_jacobian<Model::ParticleGrowth<Real2, SparseMatrix2>, Vector2, SparseMatrix2>(rxn);
    check_add_nonzero<Model::ParticleGrowth<Real2, SparseMatrix2>, Real2>(rxn);
    check_update_nonzero(rxn);
    std::cout << std::endl;
  }

  {
    const Real2 reaction_rate = 1.5;
    Model::ParticleGrowth<Real2, DenseMatrix2> rxn(B,
                                                   reaction_rate,
                                                   growth_amount,
                                                   max_particle_size,
                                                   &growth_kernel<Real2>,
                                                   reactants,
                                                   products);

    check_rhs<Model::ParticleGrowth<Real2, DenseMatrix2>, Vector2>(rxn);
    check_jacobian<Model::ParticleGrowth<Real2, DenseMatrix2>, Vector2, DenseMatrix2>(rxn);
  }
}