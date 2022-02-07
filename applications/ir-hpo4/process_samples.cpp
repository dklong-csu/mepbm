#include <src/parse_samples.h>
#include <fstream>

int main ()
{
  const auto samples_and_likelihood = MEPBM::parse_samples<double>("samples.0.txt", 6);
  const auto samples = samples_and_likelihood.first;
  const auto likelihood = samples_and_likelihood.second;

  std::ofstream samples_output;
  samples_output.open("chain.0.txt");
  for (const auto & s : samples)
  {
    for (const auto & p : s)
    {
      samples_output << p << ' ';
    }
    samples_output << std::endl;
  }
  samples_output.close();

  std::ofstream likelihood_output;
  likelihood_output.open("likelihood.0.txt");
  for (const auto & ll : likelihood)
  {
    likelihood_output << ll << std::endl;
  }
  likelihood_output.close();
}