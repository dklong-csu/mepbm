#include <iostream>
#include <string>
#include <eigen3/Eigen/Dense>
#include "models.h"
#include "histogram.h"
#include "statistics.h"
#include "data.h"

using StateVector = Eigen::VectorXd;

int main()
{
  // create data
  const Data::PomData all_data;
  const std::vector<std::vector<double>> data_diam = {all_data.tem_diam_time1, all_data.tem_diam_time2,
                                                      all_data.tem_diam_time3, all_data.tem_diam_time4};

  std::vector<std::vector<double>> data;
  for (auto vec : data_diam)
  {
    std::vector<double> tmp;
    for (auto diam : vec)
    {
      tmp.push_back(std::pow(diam/0.3000805, 3));
    }
    data.push_back(tmp);
  }

  const std::vector<double> times = {0., all_data.tem_time1, all_data.tem_time2, all_data.tem_time3, all_data.tem_time4};

  // create ODE model
  const unsigned int max_size = 2500;
  const unsigned int nucleation_order = 3;
  const double solvent = 11.3;
  const unsigned int conserved_size = 1;

  const unsigned int A_index = 0;
  const unsigned int As_index = 1;
  const unsigned int POM_index = 2;
  const unsigned int nucleation_index = 3;

  const double kf = 3.6e-2;
  const double kb = 7.27e4;
  const double k1 = 6.45e4;
  const double k2 = 1.63e4;
  const double k3 = 5.56e3;
  const double cutoff = 274;


  // Nucleation
  std::shared_ptr<Model::RightHandSideContribution> nucleation
    = std::make_shared<Model::TermolecularNucleation>(A_index, As_index, POM_index,nucleation_index,
                                                      kf, kb, k1, solvent);

  // Small Growth
  std::shared_ptr<Model::RightHandSideContribution> small_growth
    = std::make_shared<Model::Growth>(A_index, nucleation_order, cutoff, max_size,
                                      POM_index, conserved_size, k2);

  // Large Growth
  std::shared_ptr<Model::RightHandSideContribution> large_growth
    = std::make_shared<Model::Growth>(A_index, cutoff+1, max_size, max_size,
                                      POM_index, conserved_size, k3);

  // Create Model
  Model::Model three_step_alt(nucleation_order, max_size);
  three_step_alt.add_rhs_contribution(nucleation);
  three_step_alt.add_rhs_contribution(small_growth);
  three_step_alt.add_rhs_contribution(large_growth);

  // set up initial condition
  StateVector ic = StateVector::Zero(max_size+1);
  ic(0) = 0.0012;

  // set up histogram parameters
  const Histograms::Parameters hist_prm(25, 1.*nucleation_order, 1.*max_size);

  // calculate log likelihood
  const double likelihood = Statistics::log_likelihood(data, times, three_step_alt, ic, hist_prm);

  // print result
  std::cout << "log likelihood: " << likelihood;
}
