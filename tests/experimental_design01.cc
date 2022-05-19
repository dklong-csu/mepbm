#include "src/experimental_design.h"
#include <iostream>
#include <eigen3/Eigen/Core>


using Vector = Eigen::VectorXd;

/*
 * This is a test for the creation of the experimental setup of the Ir HPO4 system.
 * The intent is to ensure the conditions of the experiment are correctly encoded in the code.
 */

int main () {
  // The default constructor should make the design that matches the collected data.
  MEPBM::ExperimentalDesign<Vector, double> design;

  // Check the max particle size
  std::cout << design.max_particle_size() << std::endl;

  // Check deduced vector length
  int n_nonparticle_species = 3;
  int first_particle_size = 2;
  std::cout << design.vector_length(n_nonparticle_species, first_particle_size) << std::endl;

  // Check the initial concentrations
  std::cout << design.IC_solvent() << std::endl;
  std::cout << design.IC_precursor() << std::endl;
  std::cout << design.IC_ligand() << std::endl;

  // Check the indices for the precursor and HPO4
  std::cout << design.precursor_index() << std::endl;
  std::cout << design.ligand_index() << std::endl;

  // Check the particle index range
  std::cout << design.particle_index_range(n_nonparticle_species, first_particle_size).first << std::endl;
  std::cout << design.particle_index_range(n_nonparticle_species, first_particle_size).second << std::endl;

  // Check initial condition vector
  auto ic = design.IC_vector(n_nonparticle_species, first_particle_size);
  std::cout << *(design.get_vector_pointer(ic)) << std::endl;
}