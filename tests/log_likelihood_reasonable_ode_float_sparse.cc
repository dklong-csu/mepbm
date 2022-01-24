#include <iostream>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include "src/models.h"
#include "src/histogram.h"
#include "src/statistics.h"
#include "src/ir_pom_data.h"



using Real = float;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;



int main()
{
  // create data
  const MEPBM::PomData<Real> all_data;
  const std::vector<std::vector<Real>> data = {all_data.tem_diam_time1, all_data.tem_diam_time2,
                                               all_data.tem_diam_time3, all_data.tem_diam_time4};


  const std::vector<Real> times = {0., all_data.tem_time1, all_data.tem_time2, all_data.tem_time3, all_data.tem_time4};

  // create ODE model
  const unsigned int max_size = 2500;
  const unsigned int nucleation_order = 3;
  const Real solvent = 11.3;
  const unsigned int conserved_size = 1;

  const unsigned int A_index = 0;
  const unsigned int As_index = 1;
  const unsigned int POM_index = 2;
  const unsigned int nucleation_index = 3;

  const Real kf = 3.6e-2;
  const Real kb = 7.27e4;
  const Real k1 = 6.40e4;
  const Real k2 = 1.61e4;
  const Real k3 = 5.45e3;
  const Real cutoff = 265;


  // Nucleation
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> nucleation
      = std::make_shared<Model::TermolecularNucleation<Real, Matrix>>(A_index, As_index, POM_index,nucleation_index,
                                                                      kf, kb, k1, solvent);

  // Small Growth
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> small_growth
      = std::make_shared<Model::Growth<Real, Matrix>>(A_index, nucleation_order, cutoff, max_size,
                                                      POM_index, conserved_size, k2, nucleation_index);

  // Large Growth
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> large_growth
      = std::make_shared<Model::Growth<Real, Matrix>>(A_index, cutoff+1, max_size, max_size,
                                                      POM_index, conserved_size, k3, cutoff+1);

  // Create Model
  Model::Model<Real, Matrix> three_step_alt(nucleation_order, max_size);
  three_step_alt.add_rhs_contribution(nucleation);
  three_step_alt.add_rhs_contribution(small_growth);
  three_step_alt.add_rhs_contribution(large_growth);

  // set up initial condition
  Vector ic = Vector::Zero(max_size+1);
  ic(0) = 0.0012;

  // set up histogram parameters
  const MEPBM::Parameters<Real> hist_prm(27, 1.4, 4.1);

  // calculate log likelihood
  const Real likelihood = Statistics::log_likelihood<4, Real>(data, times, three_step_alt, ic, hist_prm);

  // print result
  std::cout << std::setprecision(20) << "log likelihood: " << likelihood;
}
