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
  std::vector<RealType> real_prm = {1, 2, 3};
  std::vector<int> int_prm = {4, 5};

  Sampling::Sample<RealType> sample(real_prm,
                                    int_prm);


  // Output
  std::cout << sample << std::endl;

}