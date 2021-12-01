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
   *    dB1/dt = -2B1^2*k*r(1)*r(1) - B1*B2*k*r(1)*r(2)                   = -300 - 375       = -675
   *    dB2/dt = B1^2*k*r(1)*r(1) - B1*B2*k*r(1)*r(2) - 2B2^2*k*r(2)*r(2) = 150 - 375 - 1875 = -2100
   *    dB3/dt = B1*B2*k*r(1)*r(2)                                        = 375              = 375
   *    dB4/dt = B2^2*k*r(2)*r(2)                                         = 937.5            = 937.5
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
   *    |dA'/dA    dA'/dL    dA'/dB_1    dA'/dB_2    dA'/dB_3    dA'/dB_4  |   |0   0  0        0     0  0|
   *    |dL'/dA    dL'/dL    dL'/dB_1    dL'/dB_2    dL'/dB_3    dL'/dB_4  |   |0   0  0        0     0  0|
   *    |dB_1'/dA  dB_1'/dL  dB_1'/dB_1  dB_1'/dB_2  dB_1'/dB_3  dB_1'/dB_4| = |0   0  -243.75  -75   0  0|
   *    |dB_2'/dA  dB_2'/dL  dB_2'/dB_1  dB_2'/dB_2  dB_2'/dB_3  dB_2'/dB_4|   |0   0  -18.75   -825  0  0|
   *    |dB_3'/dA  dB_3'/dL  dB_3'/dB_1  dB_3'/dB_2  dB_3'/dB_3  dB_3'/dB_4|   |0   0  93.75    75    0  0|
   *    |dB_4'/dA  dB_4'/dL  dB_4'/dB_1  dB_4'/dB_2  dB_4'/dB_3  dB_4'/dB_4|   |0   0  0        375   0  0|
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
  // 7 is the exact number of nonzeros but function is designed to overestimate for simplicity
  std::cout << std::boolalpha << (n_nonzero >= 7) << std::endl;
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
   *    B_i + B_j -> B_{i+j}
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

    check_rhs<Model::ParticleAgglomeration<Real1, SparseMatrix1>, Vector1>(rxn);
    check_jacobian<Model::ParticleAgglomeration<Real1, SparseMatrix1>, Vector1, SparseMatrix1>(rxn);
    check_add_nonzero<Model::ParticleAgglomeration<Real1, SparseMatrix1>, Real1>(rxn);
    check_update_nonzero(rxn);
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

    check_rhs<Model::ParticleAgglomeration<Real1, DenseMatrix1>, Vector1>(rxn);
    check_jacobian<Model::ParticleAgglomeration<Real1, DenseMatrix1>, Vector1, DenseMatrix1>(rxn);
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

    check_rhs<Model::ParticleAgglomeration<Real2, SparseMatrix2>, Vector2>(rxn);
    check_jacobian<Model::ParticleAgglomeration<Real2, SparseMatrix2>, Vector2, SparseMatrix2>(rxn);
    check_add_nonzero<Model::ParticleAgglomeration<Real2, SparseMatrix2>, Real2>(rxn);
    check_update_nonzero(rxn);
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

    check_rhs<Model::ParticleAgglomeration<Real2, DenseMatrix2>, Vector2>(rxn);
    check_jacobian<Model::ParticleAgglomeration<Real2, DenseMatrix2>, Vector2, DenseMatrix2>(rxn);
  }
}