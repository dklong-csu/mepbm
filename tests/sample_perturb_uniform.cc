#include "src/perturb_sample.h"
#include "sample.h"
#include <vector>
#include <random>
#include <iostream>
#include <utility>


using RealType = double;


int main ()
{
  std::vector<RealType> real_prm = {1, 2, 3};
  std::vector<int> int_prm = {1};
  Sampling::Sample<RealType> sample1(real_prm, int_prm);

  std::vector<RealType> perturb_mag_real = {5, 5, 5};
  std::vector<int> perturb_mag_int = {5};

  std::mt19937 rng;
  const std::uint_fast32_t random_seed = std::hash<std::string>()(std::to_string(0));
  rng.seed(random_seed);


  std::pair<Sampling::Sample<RealType>, RealType> sample_pair = Sampling::perturb_uniform(sample1, rng, perturb_mag_real, perturb_mag_int);


  // Call the same random numbers and record them
  std::mt19937 rng2;
  const std::uint_fast32_t random_seed2 = std::hash<std::string>()(std::to_string(0));
  rng2.seed(random_seed2);

  assert(perturb_mag_real.size() == real_prm.size());
  std::vector<RealType> real_rng;
  for (unsigned int i=0;i<real_prm.size();++i)
  {
    real_rng.push_back(std::uniform_real_distribution<RealType>(-perturb_mag_real[i],perturb_mag_real[i])(rng2));
  }

  assert(perturb_mag_int.size() == int_prm.size());
  std::vector<int> int_rng;
  for (unsigned int i=0; i<int_prm.size();++i)
  {
    int_rng.push_back(std::uniform_int_distribution<int>(-perturb_mag_int[i], perturb_mag_int[i])(rng2));
  }


  // Compare the random numbers to the difference between the samples
  auto sample2 = sample_pair.first;

  assert(sample2.real_valued_parameters.size() == sample1.real_valued_parameters.size());
  assert(sample2.real_valued_parameters.size() == real_rng.size());
  for (unsigned int i=0; i<real_rng.size(); ++i)
  {
    auto diff = sample2.real_valued_parameters[i]-sample1.real_valued_parameters[i];
    std::cout << std::boolalpha << (diff == real_rng[i]) << std::endl;
  }

  assert(sample2.integer_valued_parameters.size() == sample1.integer_valued_parameters.size());
  assert(sample2.integer_valued_parameters.size() == int_rng.size());
  for (unsigned int i=0; i<int_rng.size(); ++i)
  {
    auto diff = sample2.integer_valued_parameters[i]-sample1.integer_valued_parameters[i];
    std::cout << std::boolalpha << (diff == int_rng[i]) << std::endl;
  }


}