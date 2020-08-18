#include <cmath>
#include <valarray>

namespace Models
{
  // This function is a conversion between the number of atoms in a particle
  // and the number of atoms available for binding -- based off Schmidt and Smirnov's work.
  double available_atoms(const double& size)
  {
    return size * 2.677 * std::pow(size, -0.28);
  }



  // Base class to describe ODE model parameters
  class ParametersBase
  {
  public:
    virtual ~ParametersBase() = default;
  };



  // Base class to describe an ODE model
  class ModelsBase
  {
  public:
    virtual std::valarray<double> right_hand_side(const std::valarray<double>& x,
                       const ParametersBase& parameters) const = 0;
  };



  // Derived class describing:
  // A + A -> B
  // A + B -> B
  class TwoStep : public ModelsBase
  {
  public:
    class Parameters : public ParametersBase
    {
    public:
      double k1, k2;
      unsigned int w, maxsize, n_variables;

      // constructor
      Parameters(const double k1_value,
             const double k2_value,
             const unsigned int nucleation_order,
             const unsigned int maxsize_value)
      {
        k1 = k1_value;
        k2 = k2_value;
        w = nucleation_order;
        maxsize = maxsize_value;
        n_variables = maxsize - w + 2;
      }
    };

    // subroutine defining how the right hand side of the ODE is formed
    virtual std::valarray<double> right_hand_side(const std::valarray<double>& x,
                       const ParametersBase& parameters) const
    {
      const Models::TwoStep::Parameters& two_step_parameters = dynamic_cast<const Models::TwoStep::Parameters&>(parameters);

      /*
      f[0] = dn_0
      f[1] = dn_w
      f[2] = dn_{w+1}
      ...
      f[two_step_parameters.n_variable - 1) = dn_{maxsize}
      */
      std::valarray<double> f(two_step_parameters.n_variables);
      // loss from nucleation
      f[0] = -((int)two_step_parameters.w * (int)two_step_parameters.k1 * std::pow(x[0], 1. * (int)two_step_parameters.w));
      // gain from nucleation
      f[1] = two_step_parameters.k1 * std::pow(x[0], 1. * two_step_parameters.w);
      for (unsigned int i = 2; i < two_step_parameters.n_variables; i++)
      {
        // gain from growth
        f[i] = two_step_parameters.k2 * x[0] * available_atoms(two_step_parameters.w + i - 2) * x[i - 1];
        f[0] -= f[i];
      }

      for (unsigned int i = 1; i < two_step_parameters.n_variables - 1; i++)
      {
        // loss from growth
        f[i] -= f[i + 1];
      }

      // loss from growth on largest particle
      f[two_step_parameters.n_variables - 1] -= two_step_parameters.k2 * x[0] * available_atoms(two_step_parameters.maxsize) * x[two_step_parameters.n_variables - 1];
      f[0] -= two_step_parameters.k2 * x[0] * available_atoms(two_step_parameters.maxsize) * x[two_step_parameters.n_variables - 1];
      return f;
    }
  };



  // Derived class describing:
  // A + 2S <-> A_solv + POM
  // 2A_solv + A -> B + POM + S
  // A + B -> B + POM
  class TwoStepAlternative : public ModelsBase
  {
  public:
    class Parameters : public ParametersBase
    {
    public:
      double k1, k2, k_forward, k_backward, solvent;
      unsigned int w, maxsize, n_variables;

      // constructor
      Parameters(const double k_forward_value,
             const double k_backward_value,
             const double k1_value,
             const double k2_value,
             const double solvent_value,
             const unsigned int nucleation_order,
             const unsigned int maxsize_value)
      {
        k_forward = k_forward_value;
        k_backward = k_backward_value;
        k1 = k1_value;
        k2 = k2_value;
        w = nucleation_order;
        maxsize = maxsize_value;
        n_variables = maxsize - w + 4;
        solvent = solvent_value;
      }
    };

    // subroutine defining how the right hand side of the ODE is formed
    virtual std::valarray<double> right_hand_side(const std::valarray<double>& x,
                       const ParametersBase& parameters) const
    {
      const Models::TwoStepAlternative::Parameters& two_step_alt_parameters = dynamic_cast<const Models::TwoStepAlternative::Parameters&>(parameters);

      /*
      f[0] = dn_0
      f[1] = dn_s
      f[2] = dp
      f[3] = dn_w
      f[4] = dn_{w+1}
      ...
      f[two_step_alt_parameters.n_variables - 1] = dn_{maxsize}
      */
      std::valarray<double> f(two_step_alt_parameters.n_variables);
      // precursor -- loss from dissociative step, gain from dissociative step
      f[0] = -two_step_alt_parameters.k_forward * x[0] * two_step_alt_parameters.solvent * two_step_alt_parameters.solvent + two_step_alt_parameters.k_backward*x[1]*x[2];
      // dissasociated precursor -- opposite effect from the dissociative step as the precursor
      f[1] = -f[0];
      // precursor -- loss from nucleation
      f[0] -= two_step_alt_parameters.k1 * x[0] * x[1] * x[1];
      // dissasociated precursor -- loss from nucleation
      f[1] -= 2* two_step_alt_parameters.k1 * x[0] * x[1] * x[1];
      // skip ligand (p) for now since f[2] = - f[0] and we still need to update f[0]

      // nucleated particle -- gain from nucleation
      f[3] = two_step_alt_parameters.k1 * x[0] * x[1] * x[1];
      for (unsigned int i = 4; i < two_step_alt_parameters.n_variables; i++)
      {
        // particle gain from growth
        f[i] = two_step_alt_parameters.k2 * x[0] * available_atoms(i - 1) * x[i - 1];
        // precursor loss from growth
        f[0] -= f[i];
      }

      for (unsigned int i = 3; i < two_step_alt_parameters.n_variables - 1; i++)
      {
        // loss from growth
        f[i] -= f[i + 1];
      }

      // loss from growth on largest particle -- I'm torn about including this
      f[two_step_alt_parameters.n_variables - 1] -= two_step_alt_parameters.k2 * x[0] * available_atoms(two_step_alt_parameters.maxsize) * x[two_step_alt_parameters.n_variables - 1];
      f[0] -= two_step_alt_parameters.k2 * x[0] * available_atoms(two_step_alt_parameters.maxsize) * x[two_step_alt_parameters.n_variables - 1];
      // assign ligand rate now that precursor rate is final
      f[2] = -f[0];

      return f;
    }
  };


  // Derived class describing:
  // A -> B
  // A + B -> B
  // A + C -> C
  class ThreeStep : public ModelsBase
  {
  public:
    class Parameters : public ParametersBase
    {
    public:
      double k1, k2, k3;
      unsigned int w, maxsize, n_variables, particle_size_cutoff;

      // constructor
      Parameters(const double k1_value,
             const double k2_value,
             const double k3_value,
             const unsigned int nucleation_order,
             const unsigned int maxsize_value,
             const unsigned int particle_size_cutoff_value)
      {
        k1 = k1_value;
        k2 = k2_value;
        k3 = k3_value;
        w = nucleation_order;
        maxsize = maxsize_value;
        n_variables = maxsize - w + 2;
        particle_size_cutoff = particle_size_cutoff_value;
      }
    };


    // The growth rate for particles depends on their size.
    double rate_constant(const unsigned int& size,
                   const ParametersBase& parameters) const
    {
      const Models::ThreeStep::Parameters& three_step_parameters = dynamic_cast<const Models::ThreeStep::Parameters&>(parameters);
      if (size <= three_step_parameters.particle_size_cutoff)
        return three_step_parameters.k2;
      else
        return three_step_parameters.k3;
    }


    // subroutine defining how the right hand side of the ODE is formed
    virtual std::valarray<double> right_hand_side(const std::valarray<double>& x,
                       const ParametersBase& parameters) const
    {
      const Models::ThreeStep::Parameters& three_step_parameters = dynamic_cast<const Models::ThreeStep::Parameters&>(parameters);

      /*
      f[0] = dn_0
      f[1] = dn_w
      f[2] = dn_{w+1}
      ...
      f[three_step_parameters.n_variable - 1) = dn_{maxsize}
      */
      std::valarray<double> f(three_step_parameters.n_variables);
      // precursor -- loss from nucleation
      f[0] = -(int)three_step_parameters.w * three_step_parameters.k1 * std::pow(x[0], 1. * three_step_parameters.w);
      // nucleated particle -- gain from nucleation
      f[1] = three_step_parameters.k1 * std::pow(x[0], 1. * three_step_parameters.w);
      for (unsigned int i = 2; i < three_step_parameters.n_variables; i++)
      {
        // particle gain from growth
        f[i] = rate_constant(three_step_parameters.w + i - 2,three_step_parameters)*x[0]* available_atoms(three_step_parameters.w + i - 2) * x[i - 1];
        f[0] -= f[i];
      }

      for (unsigned int i = 1; i < three_step_parameters.n_variables - 1; i++)
      {
        // particle loss from growth
        f[i] -= f[i + 1];
      }

      // loss from growth on largest particle
      f[three_step_parameters.n_variables - 1] -= rate_constant(three_step_parameters.maxsize,three_step_parameters)* x[0] * available_atoms(three_step_parameters.maxsize) * x[three_step_parameters.n_variables - 1];
      f[0] -= rate_constant(three_step_parameters.maxsize, three_step_parameters) * x[0] * available_atoms(three_step_parameters.maxsize) * x[three_step_parameters.n_variables - 1];
      return f;
    }
  };


  // Derived class describing:
  // A + 2S <-> A_solv + POM
  // 2A_solv + A -> B + POM + S
  // A + B -> C + POM
  // A + C -> C + POM
  class ThreeStepAlternative : public ModelsBase
  {
  public:
    class Parameters : public ParametersBase
    {
    public:
      double k1, k2, k3, k_forward, k_backward, solvent;
      unsigned int w, maxsize, n_variables, particle_size_cutoff;

      // constructor
      Parameters(const double k_forward_value,
             const double k_backward_value,
             const double k1_value,
             const double k2_value,
             const double k3_value,
             const double solvent_value,
             const unsigned int nucleation_order,
             const unsigned int maxsize_value,
             const unsigned int particle_size_cutoff_value)
      {
        k_forward = k_forward_value;
        k_backward = k_backward_value;
        k1 = k1_value;
        k2 = k2_value;
        k3 = k3_value;
        w = nucleation_order;
        maxsize = maxsize_value;
        n_variables = maxsize - w + 4;
        particle_size_cutoff = particle_size_cutoff_value;
        solvent = solvent_value;
      }
    };

    // The growth rate for particles depends on their size.
    double rate_constant(const unsigned int& size,
      const ParametersBase& parameters) const
    {
      const Models::ThreeStepAlternative::Parameters& three_step_alt_parameters = dynamic_cast<const Models::ThreeStepAlternative::Parameters&>(parameters);
      if (size <= three_step_alt_parameters.particle_size_cutoff)
        return three_step_alt_parameters.k2;
      else
        return three_step_alt_parameters.k3;
    }

    // subroutine defining how the right hand side of the ODE is formed
    virtual std::valarray<double> right_hand_side(const std::valarray<double>& x,
      const ParametersBase& parameters) const
    {
      const Models::ThreeStepAlternative::Parameters& three_step_alt_parameters = dynamic_cast<const Models::ThreeStepAlternative::Parameters&>(parameters);

      /*
      f[0] = dn_0
      f[1] = dn_s
      f[2] = dp
      f[3] = dn_w
      f[4] = dn_{w+1}
      ...
      f[three_step_alt_parameters.n_variable - 1) = dn_{maxsize}
      */
      std::valarray<double> f(three_step_alt_parameters.n_variables);
      // precursor -- loss from dissociative step, gain from dissociative step
      f[0] = -three_step_alt_parameters.k_forward * x[0] * three_step_alt_parameters.solvent * three_step_alt_parameters.solvent + three_step_alt_parameters.k_backward * x[1]*x[2];
      // dissasociated precursor -- opposite effect from the dissociative step as the precursor
      f[1] = -f[0];
      // precursor -- loss from nucleation
      f[0] -= three_step_alt_parameters.k1 * x[0] * x[1] * x[1];
      // dissasociated precursor -- loss from nucleation
      f[1] -= 2 * three_step_alt_parameters.k1 * x[0] * x[1] * x[1];
      // skip ligand (p) for now since f[2] = - f[0] and we still need to update f[0]

      // nucleated particle -- gain from nucleation
      f[3] = three_step_alt_parameters.k1 * x[0] * x[1] * x[1];

      for (unsigned int i = 4; i < three_step_alt_parameters.n_variables; i++)
      {
        // particle gain from growth
        f[i] = rate_constant(three_step_alt_parameters.w + i - 4, three_step_alt_parameters) * x[0] * available_atoms(three_step_alt_parameters.w + i - 4) * x[i - 1];
        f[0] -= f[i];
      }

      for (unsigned int i = 3; i < three_step_alt_parameters.n_variables - 1; i++)
      {
        // loss from growth
        f[i] -= f[i + 1];
      }

      // loss from growth on largest particle -- I'm torn about including this
      f[three_step_alt_parameters.n_variables - 1] -= rate_constant(three_step_alt_parameters.maxsize,three_step_alt_parameters) * x[0] * available_atoms(three_step_alt_parameters.maxsize) * x[three_step_alt_parameters.n_variables - 1];
      f[0] -= rate_constant(three_step_alt_parameters.maxsize, three_step_alt_parameters) * x[0] * available_atoms(three_step_alt_parameters.maxsize) * x[three_step_alt_parameters.n_variables - 1];
      // assign ligand rate now that precursor rate is final
      f[2] = -f[0];

      return f;
    }
  };



  // Function to integrate the ODE
  // Explicit Euler
  std::valarray<double> integrate_ode_explicit_euler(const std::valarray<double>& x0,
                       const Models::ModelsBase& model,
                       const Models::ParametersBase& parameters,
                       const double start_time,
                       const double end_time)
  {
    std::valarray<double> x = x0;
    double time_step = 1e-6;

    double time = start_time;
    while (time <= end_time)
    {
      // advance to next time
      if (time + time_step > end_time)
      {
        time_step = end_time - time;
        time = end_time;
      }
      else
      {
        time += time_step;
      }

      // explicit euler update step
      x += time_step * model.right_hand_side(x, parameters);
    }

    return x;
  }
}
