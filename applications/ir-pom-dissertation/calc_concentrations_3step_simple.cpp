#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <numeric>

#include <omp.h>

#include "sampling_sundials.h"


using Real = realtype;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB< Matrix, Eigen::IncompleteLUT<Real> >;



double psd_mean(const Vector & particle_pmf){
  double mean = 0.0;
  for (unsigned int i=0; i<particle_pmf.size(); ++i){
    const double diam = MEPBM::atoms_to_diameter<double>(i+3);
    mean += diam * particle_pmf(i);
  }
  return mean;
}



/*==================================================================
 *  ODE solve functions
 ===================================================================*/
Real growth_kernel(const unsigned int size, const Real k)
{
  return k * (2.677 * std::pow(1.0*size, 2./3.));
}


Vector
create_old_ic(const double conc_A, const double conc_L)
{
  constexpr unsigned int max_size = 2500;
  Vector initial_condition = Vector::Zero(max_size + 1);
  initial_condition(0) = conc_A;
  initial_condition(2) = conc_L;

  return initial_condition;
}



MEPBM::ChemicalReactionNetwork<Real, Matrix>
create_ode(const Real kf, const Real kb, const Real k1, const Real k2, const Real k3, const unsigned int M, const Real Solv)
{
  constexpr unsigned int A_index = 0;
  constexpr unsigned int As_index = 1;
  constexpr unsigned int ligand_index = 2;
  constexpr unsigned int max_size = 2500;
  constexpr unsigned int conserved_size = 1;
  const Real solvent = Solv;


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
  const Real Solv = 11.3;

  auto mech = create_ode(kf,kb,k1,k2,k3,M,Solv);
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
  const Real Solv = 11.3;

  auto mech = create_ode(kf,kb,k1,k2,k3,M,Solv);
  auto jac = mech.jacobian_function();
  auto err = jac(t,x,x_dot,J,user_data,tmp1, tmp2, tmp3);
  return err;
}



// New way of solving the ODE
Vector
solve_ode(const std::vector<double> & prm, const std::vector<double> & opt_prm)
{
  const double conc_A0 = opt_prm[0];
  const double solve_time = opt_prm[1];
  const double drip_rate = opt_prm[2];
  const double drip_time = opt_prm[3];
  const double conc_L = opt_prm[4];


  auto ic_evec = create_old_ic(conc_A0, conc_L);
  auto ic = create_nvec_from_vec(ic_evec);
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(ic_evec.size(), ic_evec.size());
  auto linear_solver = MEPBM::create_sparse_iterative_solver<Matrix, Real, Solver>();

  MEPBM::CVODE<Real> ode_solver(ic, template_matrix, linear_solver,&cvode_rhs_func,&cvode_jac_func,
                                0,solve_time,5000);

  ode_solver.set_tolerance(1e-10,1e-13);
  auto prm_ptr = (void *)&prm;
  auto user_data = static_cast<void *>(prm_ptr);
  ode_solver.set_user_data(user_data);

  auto sol_pair = ode_solver.solve(solve_time);

  auto sol = sol_pair.first;
  auto s = *static_cast<Vector*>(sol->content);

  ic->ops->nvdestroy(ic);
  template_matrix->ops->destroy(template_matrix);
  linear_solver->ops->free(linear_solver);
  sol->ops->nvdestroy(sol);

  return s;
}



int main(int argc, char** argv){
  /*=============================================================
   * Import samples
   * Since this is the calculation of a cost function, the
   * evaluation needs to be consistent. As such, a number of
   * samples are drawn prior to execution of this code and
   * are the same samples used every time.
   *=============================================================*/

  std::vector< std::vector<double> > samples;
  samples.reserve(1000);

  {
    const std::string file_name = "./samples4cost.txt";
    std::ifstream infile(file_name);
    std::string line;
    while (std::getline(infile, line)){
      std::istringstream iss(line);
      std::string substring;
      std::vector< std::string > substrings;
      while (std::getline(iss, substring, ',')){
        substrings.push_back(substring);
      }

      std::vector<double> samp;
      for (auto & str : substrings){
        samp.push_back( std::stod(str) );
      }

      samples.push_back(samp);
    }
  }

  /*=============================================================
   * Import optimization parameters
   * These are parameters different from the parameters imported
   * above. Above are the reaction rate parameters from Bayesian
   * inversion. These are now the parameters we have control over
   * and want to optimize.
   *=============================================================*/

  std::vector< double > opt_prm;

  {
    std::string file_name;
    if (argc < 2)
      file_name = "./opt_prm.txt";
    else
      file_name = argv[1];
    
    std::cout << "Reading parameters from: " << file_name << "\n";
    std::ifstream infile(file_name);
    std::string line;
    while (std::getline(infile, line)){
      opt_prm.push_back( std::stod(line) );
    }
  }

  /*=============================================================
   * Solve ODEs
   * The ODEs are solved in parallel. The mean deviation from the
   * desired mean and the standard deviation of each particle
   * size distribution are kept track of.
   *=============================================================*/
  const unsigned int n_solves = samples.size();
  std::vector<Vector> psds(n_solves);

  int solves_done = 0;

  std::cout << "Solving ODE " << n_solves << " times.\n";
  std::cout << "[";
  for (unsigned int i=0;i<50;++i){
    std::cout << " ";
  }
  std::cout << "] 0%";

  std::vector<double> psd_means(n_solves);
  std::vector<double> A_conc(n_solves);
  std::vector<double> As_conc(n_solves);
  std::vector<double> L_conc(n_solves);

  #pragma omp parallel for
    for (unsigned int i=0; i<n_solves; ++i){

      const Real Solv = opt_prm[5];
      const Real kf_mult = opt_prm[7];
      const Real k1_mult = opt_prm[8];
      const Real k2_mult = opt_prm[9];
      const Real k3_mult = opt_prm[10];

      samples[i].push_back(Solv);
      samples[i].push_back(kf_mult);
      samples[i].push_back(k1_mult);
      samples[i].push_back(k2_mult);
      samples[i].push_back(k3_mult);


      auto sol = solve_ode(samples[i], opt_prm);
      const Vector particles = sol.tail(2498);
      psds[i] = particles;
      psd_means[i] = psd_mean(particles);
      A_conc[i] = sol(0);
      As_conc[i] = sol(1);
      L_conc[i] = sol(2);

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

  std::cout << "\n";


  std::string out_file_name;
  if (argc < 2)
    out_file_name = "optimal_PSD.txt";
  else {
    std::string infile = argv[1];
    out_file_name = infile.substr(0, infile.find('.')) + "_PSD.out";
  }
  std::cout << "Writing PSDs to: " << out_file_name << "\n";
  std::ofstream cost_file;
  
  cost_file.open(out_file_name);
  for (auto & v : psds){
    cost_file << std::setprecision(20) << v.transpose() << "\n";
  }
  cost_file.close();


  std::string mean_out_file_name = "mean_size_distribution.out";
  if (argc >= 2){
    std::string infile = argv[1];
    mean_out_file_name = infile.substr(0, infile.find('.')) + "_mean.out";
  }
  std::cout << "Writing mean sizes to: " << mean_out_file_name << "\n";
  std::ofstream mean_file;
  mean_file.open(mean_out_file_name);
  for (auto & v : psd_means){
    mean_file << std::setprecision(20) << v << "\n";
  }
  mean_file.close();


  const double avg_A = std::accumulate( A_conc.begin(), A_conc.end(), 0.0) / A_conc.size();
  std::cout << "Average [A] = " << avg_A << std::endl;
  std::cout << "Average [A] utilization = " << 100. * (1 - avg_A  / opt_prm[0]) << "%" << std::endl;

  const double avg_As = std::accumulate( As_conc.begin(), As_conc.end(), 0.0) / As_conc.size();
  std::cout << "Average [A_s] = " << avg_As << std::endl;
  
  const double avg_L = std::accumulate( L_conc.begin(), L_conc.end(), 0.0) / L_conc.size();
  std::cout << "Average [L] = " << avg_L << std::endl;
}