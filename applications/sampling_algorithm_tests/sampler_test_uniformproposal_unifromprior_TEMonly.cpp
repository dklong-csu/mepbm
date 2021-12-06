#include "sample.h"
#include <iostream>
#include "nvector_eigen.h"
#include <eigen3/Eigen/Sparse>
#include "sampling_algorithm.h"
#include <string>
#include "data.h"



using RealType = double;
using Matrix = Eigen::SparseMatrix<realtype, Eigen::RowMajor>;
using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;



// 3 Step mechanism for Ir-POM
// real_prm = {kf, kb, k1, k2, k3}
// int_prm = {M}
Model::Model<RealType, Matrix>
three_step(const std::vector<RealType> real_prm, const std::vector<int> int_prm)
{
  std::shared_ptr<Model::RightHandSideContribution<RealType, Matrix>> nucleation =
      std::make_shared<Model::TermolecularNucleation<RealType, Matrix>>(0, 1,2, 3, real_prm[0], real_prm[1], real_prm[2], 11.3);

  std::shared_ptr<Model::RightHandSideContribution<RealType, Matrix>> small_growth =
      std::make_shared<Model::Growth<RealType, Matrix>>(0, 3, int_prm[0],2500, 2,1, real_prm[3], 3);

  std::shared_ptr<Model::RightHandSideContribution<RealType, Matrix>> large_growth =
      std::make_shared<Model::Growth<RealType, Matrix>>(0, int_prm[0]+1, 2500,
                                                    2500, 2,
                                                    1, real_prm[4], int_prm[0]+1);
  Model::Model<RealType, Matrix> m(3,2500);
  m.add_rhs_contribution(nucleation);
  m.add_rhs_contribution(small_growth);
  m.add_rhs_contribution(large_growth);
  return m;
}


int main ()
{
  // create a sample
  std::vector<RealType> real_prm = {6e-3, 1.2e4, 1.8e5, 1.9e5, 1.7e4};
  std::vector<int> int_prm = {97};

  std::vector< std::pair<RealType, RealType> > real_prm_bounds =
      {
          {0, 1e3},
          {1000, 2e8},
          {4800, 1e8},
          {10, 1e8},
          {10, 1e8}
      };


  std::vector< std::pair<int, int> > int_prm_bounds =
      {
          {10, 2000}
      };

  std::function< Model::Model<RealType, Matrix>(const std::vector<RealType>, const std::vector<int>) > create_model_fcn
      = three_step;

  N_Vector initial_condition = create_eigen_nvector<Vector>(2501);
  auto ic_vec = static_cast<Vector*>(initial_condition->content);
  for (unsigned int i=0; i<ic_vec->size();++i)
  {
    if (i==0)
      (*ic_vec)(i) = 0.0012;
    else
      (*ic_vec)(i) = 0.;
  }
  RealType start_time = 0;
  RealType end_time = 4.838;
  RealType abs_tol = 1e-13;
  RealType rel_tol = 1e-6;

  Data::PomData<RealType> data;
  std::vector<RealType> times = {data.tem_time1, data.tem_time2, data.tem_time3, data.tem_time4};
  unsigned int first_particle_index = 3;
  unsigned int last_particle_index = 2500;

  Histograms::Parameters<RealType> binning_parameters(27,1.4,4.1);
  unsigned int first_particle_size = 3;
  unsigned int particle_size_increase = 1;
  std::vector< std::vector<RealType> > data_sets = {data.tem_diam_time1, data.tem_diam_time2, data.tem_diam_time3, data.tem_diam_time4};

  Sampling::ModelingParameters<RealType, Matrix> model_settings(real_prm_bounds,
                                                                int_prm_bounds,
                                                                create_model_fcn,
                                                                initial_condition,
                                                                start_time,
                                                                end_time,
                                                                abs_tol,
                                                                rel_tol,
                                                                times,
                                                                first_particle_index,
                                                                last_particle_index,
                                                                binning_parameters,
                                                                first_particle_size,
                                                                particle_size_increase,
                                                                data_sets);

  Sampling::Sample<RealType> sample1(real_prm,int_prm);


  // Create the sampler
  const std::vector<RealType> perturb_mag_real = {0.001, 3e2, 3e3, 3e3, 5e2};
  const std::vector<int> perturb_mag_int = {10};
  Sampling::Sampler<RealType, Matrix, Sampling::UniformProposal, Sampling::UniformPrior, Sampling::DataTEMOnly>
      sampler(sample1, perturb_mag_real, perturb_mag_int, model_settings);

  // Create a seed for the random number generator to ensure consistent results.
  const std::uint_fast32_t random_seed = std::hash<std::string>()(std::to_string(0));

  // Create an output file to print all of the samples.
  std::ofstream samples_file(std::string("samples_uniformproposal_uniformprior_TEMonly.txt"));
  sampler.generate_samples(1, samples_file, random_seed);
}