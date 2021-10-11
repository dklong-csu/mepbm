#include <valarray>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>




std::valarray<double>
line_to_valarray(std::string file_line)
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
  std::valarray<double> parameters;
  unsigned int index = 0;
  while (ss >> value)
  {
    parameters[index] = value;
    ++index;
  }

  return parameters;
}



void
update_mean(const std::valarray<double> new_sample, std::valarray<double> & mean_values, const int n_samples)
{
  assert(new_sample.size() == mean_values.size());
  for (unsigned int i=0; i<mean_values.size(); ++i)
  {
    mean_values[i] += (new_sample[i] - mean_values[i])/ n_samples;
  }
}



void
read_from_file(const std::string file_path, std::valarray<double> & mean_values, int n_samples)
{
  std::ifstream new_file;
  new_file.open(file_path);
  std::string file_line;
  while (std::getline(new_file, file_line))
  {
    auto sample = line_to_valarray(file_line);
    n_samples += 1;
    update_mean(sample, mean_values, n_samples);
  }
}



int main()
{
  const int first_file = 0;
  const int last_file = 39;

  std::valarray<double> mean_value = {0,0,0,0,0,0,0};
  int n_samples = 0;

  for (unsigned int i = first_file; i <= last_file; ++i)
  {
    std::string file_path = "samples." + std::to_string(i) + ".txt";
    read_from_file(file_path, mean_value, n_samples);
  }

  for (unsigned int i=0; i<mean_value.size(); ++i)
  {
    std::cout << mean_value[i];
    if (i < mean_value.size()-1)
      std::cout << ", ";
  }
  std::cout << std::endl;
}