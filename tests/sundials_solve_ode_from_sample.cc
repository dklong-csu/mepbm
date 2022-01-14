#include "sundials_statistics.h"
#include "sample.h"
#include <iostream>
#include <iomanip>
#include <eigen3/Eigen/Sparse>
#include <functional>
#include "src/models.h"
#include <memory>


using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SolverType = Eigen::BiCGSTAB< Matrix, Eigen::IncompleteLUT<realtype> >;


class SimpleODE : public Model::RightHandSideContribution<double, Matrix>
{
public:
  // Constructor
  SimpleODE(std::vector<double> real_prm, std::vector<int> int_prm)
    :real_prm(real_prm), int_prm(int_prm)
  {}

  // Member variables
  std::vector<double> real_prm;
  std::vector<int> int_prm;

  // Member functions
  void add_contribution_to_rhs(const Vector &x, Vector &rhs)
  {
    rhs += real_prm[0] * int_prm[0] * x;
  }

  void add_contribution_to_jacobian(const Vector &x, Matrix &jacobi)
  {
    for (unsigned int i=0; i<jacobi.rows(); ++i)
    {
      jacobi.coeffRef(i,i) += real_prm[0] * int_prm[0];
    }

    jacobi.makeCompressed();
  }

  void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<double>> &triplet_list) {}

  void update_num_nonzero(unsigned int &num_nonzero) {}
};



Model::Model<double, Matrix>
create_model(const std::vector<double> real_prm, const std::vector<int> int_prm)
{
  std::shared_ptr<Model::RightHandSideContribution<double, Matrix>> ode
    = std::make_shared<SimpleODE>(real_prm, int_prm);
  Model::Model<double, Matrix> model(0,0);
  model.add_rhs_contribution(ode);
  return model;
}




int main ()
{
  // Create sample
  std::pair<double,double> real_bound(0,1);
  std::vector< std::pair<double, double> > real_bounds(1);
  real_bounds[0] = real_bound;

  std::pair<int, int> int_bound(0,1);
  std::vector< std::pair<int, int> > int_bounds(1);
  int_bounds[0] = int_bound;

  auto ic = create_eigen_nvector<Vector>(2);
  auto ic_vec = static_cast<Vector*>(ic->content);
  *ic_vec << 1, 1;

  double start_time = 0;
  double end_time = 1;
  double abs_tol = 1e-12;
  double rel_tol = 1e-6;
  std::vector<double> times = {0.5, 1};
  unsigned int index0 = 0;
  unsigned int index1 = 1;
  Histograms::Parameters<double> prm(2, 0, 1);
  unsigned int size0 = 1;
  unsigned int increase = 1;

  std::vector<double> data0 = {1};
  std::vector<double> data1 = {1};
  std::vector< std::vector<double> > data_set(2);
  data_set[0] = data0;
  data_set[1] = data1;


  Sampling::ModelingParameters<double, Matrix> model_prm(real_bounds,
                                                         int_bounds,
                                                         &create_model,
                                                         ic,
                                                         start_time,
                                                         end_time,
                                                         abs_tol,
                                                         rel_tol,
                                                         times,
                                                         index0,
                                                         index1,
                                                         prm,
                                                         size0,
                                                         increase,
                                                         data_set);

  std::vector<double> real_prm = {10};
  std::vector<int> int_prm = {-1};
  Sampling::Sample<double> s(real_prm, int_prm);


  // Solve ODE
  auto solutions = SUNDIALS_Statistics::Internal::solve_ODE_from_sample<realtype, Matrix, SolverType, ITERATIVE>(s, model_prm);

  // Convert ODE solution to std::vector and output results
  for (auto v : solutions)
  {
    const auto y = SUNDIALS_Statistics::Internal::convert_solution_to_vector<double>(v);
    for (const auto & val : y)
    {
      std::cout << std::fixed << std::setprecision(40) << val << std::endl;
    }
  }

}