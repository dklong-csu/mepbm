#include "sample.h"
#include <iostream>
#include "nvector_eigen.h"
#include <eigen3/Eigen/Sparse>
#include <valarray>



using RealType = double;
using Matrix = Eigen::SparseMatrix<realtype, Eigen::RowMajor>;
using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;



int main ()
{
  // create a sample
  std::vector<RealType> real_prm = {1, 2, 3};
  std::vector<int> int_prm = {4, 5};



  Sampling::Sample<RealType> sample(real_prm,
                                     int_prm);


  // Convert to valarray
  auto vec = static_cast<std::valarray<RealType>>(sample);

  for (const auto val : vec)
  {
    std::cout << val << std::endl;
  }

}