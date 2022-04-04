#include "src/chemical_reaction_network.h"
#include "src/species.h"
#include "src/particle.h"
#include "src/create_nvector.h"
#include "src/create_sunmatrix.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <utility>
#include <vector>



/*
 * This tests all the member functions of ChemicalReactionNetwork
 */

using Real = realtype;
using SparseMatrix1 = Eigen::SparseMatrix<Real>;
using SparseMatrix2 = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using DenseMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using ReactionPair = std::pair<MEPBM::Species, unsigned int>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;


template<typename Input, typename Vector, typename Matrix>
void
check(Input & network)
{
  // Create the inputs
  auto x = MEPBM::create_eigen_nvector<Vector>(6);
  auto x_vec = static_cast<Vector*>(x->content);
  *x_vec << 1,2,3,4,5,6;

  auto rhs = MEPBM::create_eigen_nvector<Vector>(6);
  auto rhs_vec = static_cast<Vector*>(rhs->content);
  *rhs_vec << 0,0,0,0,0,0;

  auto J = MEPBM::create_eigen_sunmatrix<Matrix>(6,6);
  J->ops->zero(J);
  auto tmp1 = MEPBM::create_eigen_nvector<Vector>(6);
  auto tmp2 = MEPBM::create_eigen_nvector<Vector>(6);
  auto tmp3 = MEPBM::create_eigen_nvector<Vector>(6);

  // Check rhs
  auto rhs_func = network.rhs_function();
  auto err_rhs = rhs_func(0, x, rhs, nullptr);
  std::cout << err_rhs << std::endl;
  std::cout << *rhs_vec << std::endl;

  // Check Jacobian
  auto jac_func = network.jacobian_function();
  auto err_jac = jac_func(0, x, rhs, J, nullptr, tmp1, tmp2, tmp3);
  std::cout << err_jac << std::endl;

  auto J_mat = static_cast<Matrix*>(J->content);
  std::cout << *J_mat << std::endl;

  x->ops->nvdestroy(x);
  rhs->ops->nvdestroy(rhs);
  J->ops->destroy(J);
  tmp1->ops->nvdestroy(tmp1);
  tmp2->ops->nvdestroy(tmp2);
  tmp3->ops->nvdestroy(tmp3);
}



template <typename Real>
Real
growth_kernel(const unsigned int size)
{
  return size * 2.5;
}

template<typename Matrix>
void
run_test()
{
  /*
   * Test with the mechanism
   *  A + B -> C
   *  A + D -> D
   *  A + E -> E
   *  D + D -> E
   * where D, E are particles
   */
  MEPBM::Species A(0);
  MEPBM::Species B (1);
  MEPBM::Species C(2);
  MEPBM::Particle D(3,4,1);
  MEPBM::Particle E(5, 5, 3);


  // Test dense
  MEPBM::ChemicalReaction<Real, Matrix> rxn1({ {A,1}, {B,1} },
                                                  { {C,1} },
                                                  1.0);
  MEPBM::ParticleGrowth<Real, Matrix> grow1(D,
                                                 1,
                                                 3,
                                                 &growth_kernel<Real>,
                                                 {{A,1}},
                                                 {});
  MEPBM::ParticleGrowth<Real, Matrix> grow2(E,
                                                 1,
                                                 3,
                                                 &growth_kernel<Real>,
                                                 {{A,1}},
                                                 {});
  MEPBM::ParticleAgglomeration<Real, Matrix> agglom1(D,
                                                          D,
                                                          1,
                                                          3,
                                                          &growth_kernel<Real>,
                                                          {},
                                                          {});
  MEPBM::ChemicalReactionNetwork<Real, Matrix> network({rxn1},
                                                            {grow1, grow2},
                                                            {agglom1});

  check<MEPBM::ChemicalReactionNetwork<Real, Matrix>,Vector,Matrix>(network);
}


int main()
{
  run_test<DenseMatrix>();
  run_test<SparseMatrix1>();
  run_test<SparseMatrix2>();
}