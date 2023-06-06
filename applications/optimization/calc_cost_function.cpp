#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>

#include <omp.h>

#include "sampling_sundials.h"


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
  const Real Solv = prm[6];
  const Real kf_mult = prm[7];
  const Real k1_mult = prm[8];
  const Real k2_mult = prm[9];
  const Real k3_mult = prm[10];

  auto mech = create_ode(kf*kf_mult,kb,k1*k1_mult,k2*k2_mult,k3*k3_mult,M,Solv);
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
  const Real Solv = prm[6];
  const Real kf_mult = prm[7];
  const Real k1_mult = prm[8];
  const Real k2_mult = prm[9];
  const Real k3_mult = prm[10];

  auto mech = create_ode(kf*kf_mult,kb,k1*k1_mult,k2*k2_mult,k3*k3_mult,M,Solv);
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

  ode_solver.set_tolerance(1e-7,1e-18);
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

  // std::cout << "opt_prm = [";
  // for (const auto p : opt_prm){
  //   std::cout << p << std::endl;
  // }
  // std::cout << "]\n";

  /*=============================================================
   * Solve ODEs
   * The ODEs are solved in parallel. The mean deviation from the
   * desired mean and the standard deviation of each particle
   * size distribution are kept track of.
   *=============================================================*/
  const unsigned int n_solves = samples.size();
  std::vector<double> psd_means(n_solves);
  std::vector<double> psd_stds(n_solves);
  std::vector<double> A_conc(n_solves);

  #pragma omp parallel for
    for (unsigned int i=0; i<n_solves; ++i){
      // Optimization parameters Solv, kf_mult, k1_mult, k2_mult, and k3_mult need to be added to each sample
      // so the ODEs get formed properly
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
      A_conc[i] = sol(0);
      const Vector particles = sol.tail(2498);
      // const auto pmf = MEPBM::normalize_concentrations(particles);
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> pmf = MEPBM::normalize_concentrations(particles);
      double pmf_sum = 0.0;
      for (const auto & p : pmf){
        pmf_sum += p;
      }
      // std::string pmf_str = "sum of pmf = " + std::to_string(pmf_sum) + "\n";
      //std::cout << pmf_str;

      const double desired_size = opt_prm[6];
      const auto mu = psd_mean(pmf);
      const auto sigma = psd_std(pmf,mu);

      psd_means[i] = std::abs(mu - desired_size);
      // std::string diff_str = "|mean - " + std::to_string(desired_size) + "| = " + std::to_string(psd_means[i]) + "\n";
      //std::cout << diff_str;
      psd_stds[i] = sigma;
    }

  /*===================================================================
   * Compute cost function
   * Mean variation: MV = sum( size * prob(size) )
   * Particle variation: PV = sum( prob(size) * [ size - MV ]^2 )
   *  ^^^^^ calculated in the previous block
   * Cost = w1* E(MV) + w2 * E(PV) + w3 * var(MV) + w4 * var(PV)
   * E(*) = Expected value of *
   * var(*) = variance of *
   *===================================================================*/
  const double w1 = 5;
  const double w2 = 1;//1;
  const double w3 = 0;//1.e3;//1e3;
  const double w4 = 0;//1.0e4;//1/10.;
  const double w5 = 5;//5

  const double EMV = sample_mean(psd_means);
  const double EPV = sample_mean(psd_stds);
  const double vMV = sample_variance(psd_means, EMV);
  const double vPV = sample_variance(psd_stds, EPV);
  const double EAutil = sample_mean(A_conc) / opt_prm[0];

  const double cost = w1*EMV + w2*EPV + w3*vMV + w4*vPV + w5*EAutil;

  std::cout << w1 << " * " << EMV
    << " + " << w2 << " * " << EPV
    << " + " << w3 << " * " << vMV
    << " + " << w4 << " * " << vPV
    << " + " << w5 << " * " << EAutil
    << " = " << cost << std::endl;

  std::cout << "EMV proportion: " << w1*EMV/cost << "\n";
  std::cout << "EPV proportion: " << w2*EPV/cost << "\n";
  std::cout << "vMV proportion: " << w3*vMV/cost << "\n";
  std::cout << "vPV proportion: " << w4*vPV/cost << "\n";
  std::cout << "[A] regularization proportion: " << w5*EAutil/cost << "\n";


  std::string out_file_name;
  if (argc < 2)
    out_file_name = "cost_value.txt";
  else {
    std::string infile = argv[1];
    out_file_name = infile.substr(0, infile.find('.')) + ".out";
  }
  std::cout << "Writing cost to: " << out_file_name << "\n";
  std::ofstream cost_file;
  cost_file.open(out_file_name);
  cost_file << std::setprecision(20) << cost << "\n";
  cost_file.close();
  
}