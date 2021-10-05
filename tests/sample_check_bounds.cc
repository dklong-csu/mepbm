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


  Sampling::Sample<RealType> sample(real_prm,
                                     int_prm);

  // All parameters should be within bounds
  auto test1 = Sampling::sample_is_valid(sample, real_prm_bounds, int_prm_bounds);
  std::cout << std::boolalpha << test1 << std::endl;

  // Modify sample so that the real parameter is out of bounds
  std::vector<RealType> out_of_bounds_real = {3};
  sample.real_valued_parameters = out_of_bounds_real;
  auto test2 = Sampling::sample_is_valid(sample, real_prm_bounds, int_prm_bounds);
  std::cout << std::boolalpha << test2 << std::endl;

  // Modify sample so that the integer parameter is out of bounds
  std::vector<int> out_of_bounds_int = {3};
  sample.real_valued_parameters = real_prm;
  sample.integer_valued_parameters = out_of_bounds_int;
  auto test3 = Sampling::sample_is_valid(sample, real_prm_bounds, int_prm_bounds);
  std::cout << std::boolalpha << test3 << std::endl;




}