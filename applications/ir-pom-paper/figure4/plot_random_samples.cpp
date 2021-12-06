/*
 * Program to take in a set of samples, randomly choose a subset of them, and compute the particle size distribution
 * for each of the random samples. The resulting particle size distribution for each sample is exported into a Matlab
 * readable format to make visualization simple.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>
#include <algorithm>
#include <random>
#include <sstream>

#include "models.h"
#include "ode_solver.h"
#include "data.h"

// Set precision
using Real = float;

// A data type describing the linear algebra object vector that is used
// in the ODE solver.
using StateVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;

/*
 * A class to represent the parameters in a sample, simply for readability.
 */
struct Sample
{
  Sample();
  Sample(Real kf, Real kb, Real k1, Real k2, Real k3, unsigned int cutoff);
  Sample(Real kf, Real kb, Real k1, Real k2, Real k3, Real k4, unsigned int cutoff);

  Real kf, kb, k1, k2, k3, k4;
  unsigned int cutoff;
  bool is_4step;

  // What it means to output a Sample
  friend std::ostream & operator<< (std::ostream &out, const Sample &sample);
};



Sample::Sample() {};



Sample::Sample(Real kf, Real kb, Real k1, Real k2, Real k3, unsigned int cutoff)
  : kf(kf), kb(kb), k1(k1), k2(k2), k3(k3), k4(0), cutoff(cutoff), is_4step(false)
{};



Sample::Sample(Real kf, Real kb, Real k1, Real k2, Real k3, Real k4, unsigned int cutoff)
    : kf(kf), kb(kb), k1(k1), k2(k2), k3(k3), k4(k4), cutoff(cutoff), is_4step(true)
{};



// When printing to the terminal or to a file, we want comma delimited data where columns
// refer to parameters and rows refer to samples
std::ostream &operator<<(std::ostream &out, const Sample &sample)
{
  out << sample.kf << ", "
      << sample.kb << ", "
      << sample.k1 << ", "
      << sample.k2 << ", "
      << sample.k3 << ", "
      << sample.k4 << ", "
      << sample.cutoff ;
  return out;
}



/*
 * A function that converts a comma-delimited string into a sample
 * str -- a comma-delimited string containing numerical parameter values
 */
Sample string_to_sample(std::string str)
{
  // Remove all commas from the string and replace with white space
  std::size_t position = 0;
  while (position < str.size())
  {
    if ( (position = str.find_first_of(',',position)) != std::string::npos)
      str[position] = ' ';
  }

  // Use a string stream to move through the string and extract the numbers
  std::stringstream ss(str);
  Real d = 0;
  std::vector<Real> sample_vector;
  while (ss >> d)
    sample_vector.push_back(d);

  // Use the vector to create a sample
  const int num_params = sample_vector.size();

  switch (num_params)
  {
    case 6:
      return Sample(sample_vector[0],
                    sample_vector[1],
                    sample_vector[2],
                    sample_vector[3],
                    sample_vector[4],
                    (unsigned int)sample_vector[5]);
    case 7:
      return Sample(sample_vector[0],
                    sample_vector[1],
                    sample_vector[2],
                    sample_vector[3],
                    sample_vector[4],
                    sample_vector[5],
                    (unsigned int)sample_vector[6]);
    default:
      std::cerr << std::endl
                << std::endl
                << "------------------------------------------"
                << std::endl
                << "Exception on processing: " << std::endl
                << "Sample does not have 6 or 7 parameters. This is incompatible with the 3- or 4-step."
                << std::endl
                << "Aborting!"
                << std::endl
                << "------------------------------------------"
                << std::endl;
  }

  return Sample();
}



/*
 * A function that opens a file, converts each line of the file into a sample, and adds that sample to a vector
 * file_name -- file to open
 * samples -- a vector containing the samples
 */
void read_sample_file(const std::string file_path, std::vector<Sample> & samples )
{

  std::ifstream new_file;
  new_file.open(file_path);
  std::string file_line;
  while (std::getline(new_file, file_line))
  {
    Sample s = string_to_sample(file_line);
    samples.push_back(s);
  }
  new_file.close();
}



/*
 * A function that takes a sample and outputs an ODE model
 */
Model::Model<Real, Matrix> sample_to_model(const Sample & sample)
{
  // Hyperparameters that are the same for every model
  constexpr unsigned int A_index = 0;
  constexpr unsigned int As_index = 1;
  constexpr unsigned int ligand_index = 2;
  constexpr unsigned int min_size = 3;
  constexpr unsigned int max_size = 2500;
  constexpr unsigned int conserved_size = 1;
  constexpr Real solvent = 11.3;

  // Create the model
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> nucleation =
      std::make_shared<Model::TermolecularNucleation<Real, Matrix>>(A_index, As_index,
                                                                    ligand_index, min_size,
                                                                    sample.kf, sample.kb, sample.k1, solvent);

  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> small_growth =
      std::make_shared<Model::Growth<Real, Matrix>>(A_index, min_size, sample.cutoff,
                                                    max_size, ligand_index,
                                                    conserved_size, sample.k2, min_size);

  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> large_growth =
      std::make_shared<Model::Growth<Real, Matrix>>(A_index, sample.cutoff+1, max_size,
                                                    max_size, ligand_index,
                                                    conserved_size, sample.k3, sample.cutoff+1);


  Model::Model<Real, Matrix> model(min_size, max_size);
  model.add_rhs_contribution(nucleation);
  model.add_rhs_contribution(small_growth);
  model.add_rhs_contribution(large_growth);

  if (sample.is_4step)
  {
    std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> agglomeration =
        std::make_shared<Model::Agglomeration<Real, Matrix>>(min_size, sample.cutoff,
                                                             min_size, sample.cutoff,
                                                             max_size, conserved_size,
                                                             sample.k4, min_size);
    model.add_rhs_contribution(agglomeration);
  }

  return model;
}



/*
 * A function to create the initial condition for the ODE
 */
StateVector create_initial_condition()
{
  constexpr unsigned int max_size = 2500;
  StateVector initial_condition = StateVector::Zero(max_size + 1);
  initial_condition(0) = 0.0012;

  return initial_condition;
}



/*
 * A function that takes on ODE model and outputs the solutions at a few time points
 */
std::vector< StateVector > compute_solutions(const Model::Model<Real, Matrix> & model, const Real dt)
{
  ODE::StepperBDF<4, Real, Matrix> stepper(model);
  std::vector< StateVector > solutions;
  solutions.push_back( create_initial_condition() );
  const Data::PomData<Real> tem_data;
  const std::vector<Real> times = {0., tem_data.tem_time1, tem_data.tem_time2, tem_data.tem_time3, tem_data.tem_time4};

  for (unsigned int i = 1; i < times.size()+1; ++i)
  {
    auto solution = ODE::solve_ode<Real>(stepper, solutions[i-1], times[i-1], times[i], dt);
    solutions.push_back(solution);
  }

  return solutions;
}



/*
 * A function to format a solution vector for Matlab
 */
void export_to_matlab(const std::string file_path,
                      const std::string matlab_variable_name,
                      const StateVector & solution )
{
  std::ofstream file(file_path, std::ios::app);

  file << std::endl
       << matlab_variable_name
       << " = ["
       << solution
       << "\n];";

  file.close();
}



int main() {


  // Folders containing the samples
  const std::string in_dir = "/home/danny/research/ir_pom_bayesian_inverse/pom_samples_const_kdiff/";
  const std::vector<std::string> folders = { "all_3step", "all_4step"};
  const std::string out_dir = "/home/danny/p/repos/mepbm/applications/ir-pom-paper/figure4/";

  // First and last chain number
  constexpr int first_chain = 0;
  constexpr int last_chain = 231;

  // Number of samples in each chain
  constexpr int samples_per_chain = 20000;

  // Number of samples to choose
  constexpr int samples_to_choose = 100;
  constexpr int num_total_samples = (last_chain - first_chain + 1) * samples_per_chain;

  // Number of output files
  constexpr int num_files = 4;

  for (const auto & folder : folders)
  {
    std::vector<Sample> samples;
    samples.reserve(num_total_samples);

    for (auto i = first_chain; i < last_chain + 1; ++i)
    {
      const std::string file = in_dir + folder + "/samples." + std::to_string(i) + ".txt";
      std::cout << "Reading: "
                << file
                << "...";
      read_sample_file(file, samples);
      std::cout << "done" << std::endl;
    }

    std::cout << "Selecting "
              << samples_to_choose
              << " random samples...";

    // Shuffle the vector of samples and keep the first <samples_to_choose> of them
    const std::uint_fast32_t random_seed
        = std::hash<std::string>()(std::to_string(0));

    std::mt19937 rng;
    rng.seed(random_seed);

    std::shuffle(samples.begin(), samples.end(), rng);

    // Samples 0...samples_to_choose are random samples
    // Sample samples_to_choose+1 is the mean value
    std::vector<Sample> random_samples(samples_to_choose+1);
    for (unsigned int i = 0; i < samples_to_choose; ++i)
    {
      random_samples[i] = samples[i];
    }

    std::cout << "done" << std::endl;

    std::cout << "Calculating mean parameter values...";
    int mean_counter = 1;
    std::vector<Real> mean_vals(7, 0);
    for (const auto & sample : samples)
    {
      mean_vals[0] += (sample.kf - mean_vals[0]) / mean_counter;
      mean_vals[1] += (sample.kb - mean_vals[1]) / mean_counter;
      mean_vals[2] += (sample.k1 - mean_vals[2]) / mean_counter;
      mean_vals[3] += (sample.k2 - mean_vals[3]) / mean_counter;
      mean_vals[4] += (sample.k3 - mean_vals[4]) / mean_counter;
      mean_vals[5] += (sample.k4 - mean_vals[5]) / mean_counter;
      mean_vals[6] += (sample.cutoff - mean_vals[6]) / mean_counter;
      ++mean_counter;
    }
    random_samples[samples_to_choose] = Sample(mean_vals[0],
                                               mean_vals[1],
                                               mean_vals[2],
                                               mean_vals[3],
                                               mean_vals[4],
                                               mean_vals[5],
                                               (unsigned int)mean_vals[6]);
    if (mean_vals[5] <= 0.)
      random_samples[samples_to_choose].is_4step = false;

    std::cout << "done" << std::endl;

    // Write data to a matlab file
    std::cout << "Creating files...";
    std::vector<std::string> files(num_files);
    for (unsigned int t = 0; t < num_files; ++t)
    {
      std::string file_name = out_dir + folder + "/random_solutions_time" + std::to_string(t + 1) + ".m";
      files[t] = file_name;
      std::ofstream file(file_name);

      file << "solutions = cell(" << samples_to_choose << ",1);" << std::endl;
      file.close();

    }
    std::cout << "done" << std::endl;

    int counter = 1;
    for (auto &sample : random_samples) {
      std::cout << "Solving ODE " << counter << "..." << std::flush;
      const auto model = sample_to_model(sample);
      const auto solutions = compute_solutions(model, 5.e-3);
      std::cout << "done" << std::flush << std::endl;

      for (unsigned int t = 0; t < num_files; ++t) {
        const std::string matlab_var_name =
            "solutions{" + std::to_string(counter) + "}";
        export_to_matlab(files[t], matlab_var_name, solutions[t+1]);
      }
      ++counter;
    }

    std::cout << "Matlab files created!" << std::endl << std::endl;
  }



  std::cout << "\nUsing optimal deterministic parameters\n";
  const Sample deterministic_sample(3.6e-2, 7.27e4, 6.55e4, 1.65e4, 5.63e3, 274);
  const auto model = sample_to_model(deterministic_sample);
  std::cout << "Solving ODE..." << std::flush;
  const auto solutions = compute_solutions(model, 5.e-3);
  std::cout << "done" << std::flush << std::endl;

  const std::string file_name = out_dir + "deterministic/solutions.m";
  for (unsigned int t = 0; t < num_files; ++t)
  {
    const std::string matlab_var_name = "sol_t" + std::to_string(t+1);
    export_to_matlab(file_name, matlab_var_name, solutions[t+1]);
  }

  std::cout << "\nAll Matlab files created.\n";
  return 0;
}

