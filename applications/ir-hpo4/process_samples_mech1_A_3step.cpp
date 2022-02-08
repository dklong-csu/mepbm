#include <src/parse_samples.h>
#include <fstream>

int main ()
{
  const std::string file_root = "/home/danny/r/mepbm/ir-hpo4/mech1_A_3step/";

  const unsigned int first_file = 0;
  const unsigned int last_file = 95;
  for (unsigned int i=first_file; i<=last_file; ++i)
  {
    const std::string file_name = file_root + "samples." + std::to_string(i) + ".txt";
    const auto samples_and_likelihood = MEPBM::parse_samples<double>(file_name, 6);
    const auto samples = samples_and_likelihood.first;
    const auto likelihood = samples_and_likelihood.second;

    std::ofstream samples_output;
    const std::string chain_file = file_root + "chain." + std::to_string(i) + ".txt";
    samples_output.open(chain_file);
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
    const std::string likelihood_file = file_root + "likelihood." + std::to_string(i) + ".txt";
    likelihood_output.open(likelihood_file);
    for (const auto & ll : likelihood)
    {
      likelihood_output << ll << std::endl;
    }
    likelihood_output.close();
  }
  std::cout << "Files " << first_file << "-" << last_file << " processed." << std::endl;
}