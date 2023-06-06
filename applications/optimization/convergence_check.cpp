#include "sampling_sundials.h"
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <omp.h>


using Real = realtype;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB< Matrix, Eigen::IncompleteLUT<Real> >;



/*==================================================================
 * Calculation of cost function  
 ===================================================================*/
double psd_mean(const Vector & particle_pmf){
  double mean = 0.0;
  for (unsigned int i=0; i<particle_pmf.size(); ++i){
    const double diam = MEPBM::atoms_to_diameter<double>(i+3);
    mean += diam * particle_pmf(i);
  }
  return mean;
}



double psd_std(const Vector & particle_pmf, const double psd_mean){
  double var = 0.0;
  for (unsigned int i=0; i<particle_pmf.size(); ++i){
    const double diam = MEPBM::atoms_to_diameter<double>(i+3);
    var += particle_pmf(i) * ( diam - psd_mean) * (diam - psd_mean);
  }
  return std::sqrt(var);
}



double sample_mean(const std::vector<double> & samps){
  double mu = 0.0;
  const int N = samps.size();
  for (unsigned int i=0; i<N; ++i){
    mu += samps[i] / N;
  }
  return mu;
}



double sample_variance(const std::vector<double> & samps, const double sample_mean){
  double var = 0.0;
  const int N = samps.size();
  for (unsigned int i=0; i<N; ++i){
    var += ( samps[i] - sample_mean ) * ( samps[i] - sample_mean ) / (N-1);
  }
  return var;
}



/*==================================================================
 *  ODE solve functions
 ===================================================================*/
Real growth_kernel(const unsigned int size, const Real k)
{
  return k * (1.0 * size) * (2.677 * std::pow(1.0*size, -0.28));
}


Vector
create_old_ic()
{
  constexpr unsigned int max_size = 2500;
  Vector initial_condition = Vector::Zero(max_size + 1);
  initial_condition(0) = 0.0012;

  return initial_condition;
}



MEPBM::ChemicalReactionNetwork<Real, Matrix>
create_ode(const Real kf, const Real kb, const Real k1, const Real k2, const Real k3, const unsigned int M)
{
  constexpr unsigned int A_index = 0;
  constexpr unsigned int As_index = 1;
  constexpr unsigned int ligand_index = 2;
  constexpr unsigned int max_size = 2500;
  constexpr unsigned int conserved_size = 1;
  constexpr Real solvent = 11.3;


  MEPBM::Species A(A_index);
  MEPBM::Species As(As_index);
  MEPBM::Species L(ligand_index);
  MEPBM::Particle B(3,M,3);
  MEPBM::Particle C(M+1,max_size,M+1);

  MEPBM::ChemicalReaction<Real, Matrix> rxn1({ {A,1} },
                                             { {As,1}, {L,1} },
                                             solvent*solvent*kf);
  MEPBM::ChemicalReaction<Real, Matrix> rxn2({ {As, 1}, {L,1} },
                                             { {A,1} },
                                             kb);
  auto B_nuc = B.species(3);
  MEPBM::ChemicalReaction<Real, Matrix> rxn3({{As,2}, {A,1}},
                                             {{B_nuc,1}, {L,1}},
                                             k1);
  MEPBM::ParticleGrowth<Real, Matrix> rxn4(B,conserved_size,max_size,[&](const unsigned int size){return growth_kernel(size, k2);},{{A,1}},{{L,1}});
  MEPBM::ParticleGrowth<Real, Matrix> rxn5(C, conserved_size, max_size, [&](const unsigned int size){return growth_kernel(size, k3);}, {{A,1}},{{L,1}});

  MEPBM::ChemicalReactionNetwork<Real, Matrix> mech({rxn1,rxn2,rxn3},{rxn4,rxn5},{});
  return mech;
}



// Convert an Eigen vector to a N_Vector
N_Vector
create_nvec_from_vec(const Vector & e_vec)
{
  auto n_vec = MEPBM::create_eigen_nvector<Vector>(e_vec.size());
  auto v = static_cast<Vector*>(n_vec->content);
  for (unsigned int i=0;i<v->size();++i)
  {
    (*v)(i) = e_vec(i);
  }
  return n_vec;
}


// rhs function for ODE solver using mean value parameters
int cvode_rhs_func(Real t, N_Vector x, N_Vector x_dot, void * user_data)
{
  auto prm = *static_cast<std::vector<double>*>(user_data);
  // values used
  const Real kf = prm[0];
  const Real kb = prm[1];
  const Real k1 = prm[2];
  const Real k2 = prm[3];
  const Real k3 = prm[4];
  const unsigned int M = prm[5];
  // std::cout << "rhs prm: "
  //           << kf << "    "
  //           << kb << "    "
  //           << k1 << "    "
  //           << k2 << "    "
  //           << k3 << "    "
  //           << M
  //           << std::endl;

  auto mech = create_ode(kf,kb,k1,k2,k3,M);
  auto rhs = mech.rhs_function();
  auto err = rhs(t,x,x_dot,user_data);
  return err;
}



// jacobian function for ODE solver using mean value parameters
int cvode_jac_func(Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void * user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  auto prm = *static_cast<std::vector<double>*>(user_data);
  // values used
  const Real kf = prm[0];
  const Real kb = prm[1];
  const Real k1 = prm[2];
  const Real k2 = prm[3];
  const Real k3 = prm[4];
  const unsigned int M = prm[5];

  // std::cout << "jac prm: "
  //           << kf << "    "
  //           << kb << "    "
  //           << k1 << "    "
  //           << k2 << "    "
  //           << k3 << "    "
  //           << M
  //           << std::endl;

  auto mech = create_ode(kf,kb,k1,k2,k3,M);
  auto jac = mech.jacobian_function();
  auto err = jac(t,x,x_dot,J,user_data,tmp1, tmp2, tmp3);
  return err;
}



// New way of solving the ODE
Vector
solve_ode(const Real t, const std::vector<double> & prm)
{
  auto ic_evec = create_old_ic();
  auto ic = create_nvec_from_vec(ic_evec);
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(ic_evec.size(), ic_evec.size());
  auto linear_solver = MEPBM::create_sparse_iterative_solver<Matrix, Real, Solver>();
  MEPBM::CVODE<Real> ode_solver(ic, template_matrix, linear_solver,&cvode_rhs_func,&cvode_jac_func,0,5,5000);
  ode_solver.set_tolerance(1e-7,1e-13);
  auto prm_ptr = (void *)&prm;
  auto user_data = static_cast<void *>(prm_ptr);
  ode_solver.set_user_data(user_data);
  auto sol_pair = ode_solver.solve(t);
  auto sol = sol_pair.first;
  auto s = *static_cast<Vector*>(sol->content);

  ic->ops->nvdestroy(ic);
  template_matrix->ops->destroy(template_matrix);
  linear_solver->ops->free(linear_solver);
  sol->ops->nvdestroy(sol);

  return s;
}



int main (int argc, char** argv)
{
  if (argc < 2){
    throw std::runtime_error(std::string("Failed: You need to provide the number of samples to solve!\n"));
  }
  // Import all samples
  auto start_import = std::chrono::high_resolution_clock::now();

  std::vector< std::vector<double> > samples;
  const unsigned int n_solves = std::atoi(argv[1]);
  samples.reserve(n_solves);

  std::cout << "Importing samples\n" << std::flush;
  // Import files
  const std::string file_name = "./samples4cost.txt";
  std::ifstream infile(file_name);
  std::string line;
  int line_num = 0;
  while (std::getline(infile, line)){
    std::istringstream iss(line);
    std::string substring;
    std::vector<std::string> substrings;
    while (std::getline(iss, substring, ',')){
      substrings.push_back(substring);
    }

    std::vector<double> samp;
    for (auto & str: substrings){
      samp.push_back(std::stod(str));
    }

    samples.push_back(samp);
    ++line_num;
  }


  auto stop_import = std::chrono::high_resolution_clock::now();
  auto import_duration = std::chrono::duration_cast<std::chrono::seconds>(stop_import - start_import);
  std::cout << "Sample import time: " << import_duration.count() << " seconds." << std::endl;
  std::cout << samples.size() << " samples imported.\n";


  // Setup and solve ODE for each sample, extracting particle PMF for each

  unsigned int n_threads = 1;
  #ifdef _OPENMP
    n_threads = omp_get_max_threads();
  #endif

  std::cout << "Running with " << n_threads << " thread(s).\n";

  // const int n_solves = 1000000;
  std::vector<double> solve_times(n_solves);
  std::vector<double> psd_means(n_solves);
  std::vector<double> psd_stds(n_solves); 
  std::vector<double> aconv(n_solves);

  auto start_solves = std::chrono::high_resolution_clock::now();
  int solves_done = 0;

  std::cout << "Solving ODE " << n_solves << " times.\n";
  std::cout << "[";
  for (unsigned int i=0;i<50;++i){
    std::cout << " ";
  }
  std::cout << "] 0%";

  #pragma omp parallel for
    for (unsigned int i=0; i<n_solves; ++i){
      auto start_ode = std::chrono::high_resolution_clock::now();
      auto sol = solve_ode(4.838, samples[i]);
      auto stop_ode = std::chrono::high_resolution_clock::now();
      auto ode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_ode - start_ode);
      const double t = ode_duration.count();
      solve_times[i] = t;
      //std::cout << "ODE solve time: " << ode_duration.count() << " seconds." << std::endl;
      const Vector particles = sol.tail(2498);
      const auto pmf = MEPBM::normalize_concentrations(particles);

      const double desired_size = MEPBM::atoms_to_diameter<double>(750);
      const auto mu = psd_mean(pmf);
      const auto sigma = psd_std(pmf, mu);
      
      psd_means[i] = std::abs(mu - desired_size);
      psd_stds[i] = sigma;
      const auto Aconc = sol(0);
      aconv[i] = Aconc / 0.0012;

      #pragma omp critical
      {
        ++solves_done;
        std::cout << "\r";
        const auto p = (100. * solves_done) / n_solves;
        std::cout << "[";
        for (unsigned int j=0; j<50; ++j){
          if (j*100/50 <= p)
            std::cout << "#";
          else
            std::cout << " ";
        }
        std::cout << "]";
        std::cout << (int)p << "%" << std::flush;
      }

    }

  auto stop_solves = std::chrono::high_resolution_clock::now();
  auto solves_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_solves - start_solves);
  std::cout << "\nTime to solve all ODEs: " << solves_duration.count()/1000. << " seconds." << std::endl;

  double mean_solve_time = 0;
  for (unsigned int i=0; i<solve_times.size(); ++i){
    mean_solve_time += (solve_times[i] - mean_solve_time) / (i+1);
  }

  std::cout << "Average solve time: " << mean_solve_time/1000 << "s.\n";

  // Increment the value of each contribution to the cost function
  std::vector<double> moving_avg_EMV(n_solves);
  std::vector<double> moving_avg_EPV(n_solves);
  std::vector<double> moving_avg_EAC(n_solves);
  std::vector<double> moving_avg_varMV(n_solves-1);
  std::vector<double> moving_avg_varPV(n_solves-1);

  std::ofstream EMV_file;
  EMV_file.open("moving_avg_EMV.txt");
  std::ofstream EPV_file;
  EPV_file.open("moving_avg_EPV.txt");
  std::ofstream varMV_file;
  varMV_file.open("moving_avg_varMV.txt");
  std::ofstream varPV_file;
  varPV_file.open("moving_avg_varPV.txt");
  std::ofstream EAC_file;
  EAC_file.open("moving_avg_EAC.txt");
  for (unsigned int i=0; i<n_solves; ++i){
    std::vector<double>::const_iterator first_mu = psd_means.begin();
    std::vector<double>::const_iterator last_mu = psd_means.begin() + i + 1;
    std::vector<double> increment_means(first_mu, last_mu);


    std::vector<double>::const_iterator first_std = psd_stds.begin();
    std::vector<double>::const_iterator last_std = psd_stds.begin() + i + 1;
    std::vector<double> increment_std(first_std, last_std);

    std::vector<double>::const_iterator first_ac = aconv.begin();
    std::vector<double>::const_iterator last_ac = aconv.begin() + i + 1;
    std::vector<double> increment_ac(first_ac, last_ac);

    moving_avg_EMV[i] = sample_mean(increment_means);
    moving_avg_EPV[i] = sample_mean(increment_std);
    moving_avg_EAC[i] = sample_mean(increment_ac);
    if (i>0){
      moving_avg_varMV[i-1] = sample_variance(increment_means, moving_avg_EMV[i]);
      moving_avg_varPV[i-1] = sample_variance(increment_std, moving_avg_EPV[i]);
      varMV_file << std::setprecision(20) << moving_avg_varMV[i-1] << "\n";
      varPV_file << std::setprecision(20) << moving_avg_varPV[i-1] << "\n";
    }

    EMV_file << std::setprecision(20) << moving_avg_EMV[i] << "\n";
    EPV_file << std::setprecision(20) << moving_avg_EPV[i] << "\n";
    EAC_file << std::setprecision(20) << moving_avg_EAC[i] << "\n";
    

  }

  EMV_file.close();
  EPV_file.close();
  varMV_file.close();
  varPV_file.close();
  EAC_file.close();
}