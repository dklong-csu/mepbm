#include <valarray>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <eigen3/Eigen/Dense>




Eigen::Matrix<double, Eigen::Dynamic, 1>
line_to_vector(std::string file_line)
{
  // Remove all commas from the string and replace with white space
  std::size_t position = 0;
  while (position < file_line.size())
  {
    if ( (position = file_line.find_first_of(',',position)) != std::string::npos)
      file_line[position] = ' ';
  }

  // Use a string stream to move through the string and extract the numbers
  std::stringstream ss(file_line);
  double value = 0;
  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters(7);
  unsigned int index = 0;
  while (ss >> value)
  {
    parameters(index) = value;
    ++index;
  }

  return parameters;
}



void
update_mean(const Eigen::Matrix<double, Eigen::Dynamic, 1> new_sample, Eigen::Matrix<double, Eigen::Dynamic, 1> & mean_values, const int n_samples)
{
  assert(new_sample.size() == mean_values.size());
  mean_values += (new_sample - mean_values) / n_samples;
}



void
update_covariance(const Eigen::Matrix<double, Eigen::Dynamic, 1> new_sample,
                  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> & covariance,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> mean_value,
                  const int n_samples)
{
  if (n_samples > 1) {
    auto dim = new_sample.size();
    for (unsigned int i=0; i<dim; ++i)
    {
      const auto delta_i = new_sample(i) - mean_value(i);
      for (unsigned int j=0; j<dim; ++j)
      {
        const auto delta_j = new_sample(j) - mean_value(j);
        covariance(i,j) += ( (delta_i*delta_j) )/(1.0*n_samples) - covariance(i,j)/( 1.0*n_samples - 1);
      }
    }
  }
}



void
read_from_file(const std::string file_path,
               Eigen::Matrix<double, Eigen::Dynamic, 1>& mean_values,
               Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> & covariance,
               int n_samples)
{
  std::ifstream new_file;
  new_file.open(file_path);
  std::string file_line;
  while (std::getline(new_file, file_line))
  {
    auto sample = line_to_vector(file_line);
    ++n_samples;
    update_covariance(sample, covariance, mean_values, n_samples);
    update_mean(sample, mean_values, n_samples);
  }
}



int main(int argc, char **argv)
{
  int first_file = 0;
  int last_file = 0;
  // if argc == 1 then no command line input was given, so assume first and last file correspond to chain 0
  if (argc == 2)
  {
    // This means one input was given to the command line. Assume this means the input is a chain number and only that chain should be analyzed.
    first_file = atoi(argv[1]);
    last_file = atoi(argv[1]);
  }
  else if (argc > 2)
  {
    // This means more than one input was given. Assume the first is the first chain and the second is the last chain.
    // If more arguments are given, ignore them.
    first_file = atoi(argv[1]);
    last_file = atoi(argv[2]);
  }

  Eigen::Matrix<double, Eigen::Dynamic, 1> mean_value(7);
  mean_value << 0,0,0,0,0,0,0;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> covariance = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(7,7);
  int n_samples = 0;

  for (unsigned int i = first_file; i <= last_file; ++i)
  {
    std::string file_path = "samples." + std::to_string(i) + ".txt";
    read_from_file(file_path, mean_value, covariance, n_samples);
  }

  std::cout << "Mean value:\n";
  for (unsigned int i=0; i<mean_value.size(); ++i)
  {
    std::cout << mean_value(i) << ", ";
  }
  std::cout << std::endl;
  std::cout << "Covariance:\n";
  for (unsigned int i=0; i<mean_value.size(); ++i)
  {
    for (unsigned int j=0; j<mean_value.size(); ++j)
    {
      std::cout << covariance(i,j) << ", ";
    }
    std::cout << std::endl;
  }
}