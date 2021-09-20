#include "sample.h"
#include <iostream>
#include "nvector_eigen.h"
#include <eigen3/Eigen/Sparse>



using RealType = double;
using Matrix = Eigen::SparseMatrix<realtype, Eigen::RowMajor>;
using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

// A dummy function to create a Model
Model::Model<RealType, Matrix>
fcn(const std::vector<RealType>, const std::vector<int>)
{
  Model::Model<RealType, Matrix> m(1,1);
  return m;
}


int main ()
{
  // create a sample
  std::vector<RealType> real_prm = {1};
  std::vector<int> int_prm = {1};

  std::pair<RealType, RealType> real_bound(0,2);
  std::vector< std::pair<RealType, RealType> > real_prm_bounds(1);
  real_prm_bounds[0] = real_bound;

  std::pair<int, int> int_bound(0,2);
  std::vector< std::pair<int, int> > int_prm_bounds(1);
  int_prm_bounds[0] = int_bound;

  std::function< Model::Model<RealType, Matrix>(const std::vector<RealType>, const std::vector<int>) > create_model_fcn
      = fcn;

  N_Vector initial_condition = create_eigen_nvector<Vector>(1);
  RealType start_time = 0;
  RealType end_time = 1;
  RealType abs_tol = 1e-13;
  RealType rel_tol = 1e-6;

  std::vector<RealType> times(1);
  unsigned int first_particle_index = 0;
  unsigned int last_particle_index = 0;

  Histograms::Parameters<RealType> binning_parameters(1,0,1);
  unsigned int first_particle_size = 1;
  unsigned int particle_size_increase = 0;
  std::vector< std::vector<RealType> > data(1);

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
                                                                data);

  Sampling::Sample<RealType, Matrix> sample1(real_prm,
                                             int_prm,
                                             model_settings);


  // Modify the parameters
  std::vector<RealType> new_prm_real = {1,2,3};
  std::vector<int> new_prm_int = {4,5};
  auto sample2 = sample1.create_new_sample(new_prm_real, new_prm_int);

  for (auto & val : sample2.real_valued_parameters)
    std::cout << val << std::endl;

  for (auto & val : sample2.integer_valued_parameters)
    std::cout << val << std::endl;
}