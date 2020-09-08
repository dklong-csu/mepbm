#ifndef MODELS_H_
#define MODELS_H_

#include <cmath>
#include <ostream>
#include <valarray>
#include <vector>



namespace Models
{
  // This function is a conversion between the number of atoms in a particle
  // and the number of atoms available for binding -- based off Schmidt and Smirnov's work.
  double available_atoms(const double& size);



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



    virtual double returnConcentration(const std::valarray<double>& particleSizeDistribution,
                                       const unsigned int& particleSize,
                                       const ParametersBase& parameters) const = 0;


    virtual unsigned int particleSizeToIndex(const unsigned int& particleSize,
                                                const ParametersBase& parameters) const = 0;



    virtual unsigned int getSmallestParticleSize(const ParametersBase& parameters) const = 0;



    virtual unsigned int getLargestParticleSize(const ParametersBase& parameters) const = 0;
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

      // default constructor. creates an invalid object
      Parameters();

      // constructor
      Parameters(const double k1_value,
             const double k2_value,
             const unsigned int nucleation_order,
             const unsigned int maxsize_value);
             
      // Setting parameters equal, skipping the const members
      Parameters operator = (const Parameters &prm);
      
      friend
      std::ostream &
      operator<< (std::ostream &out,
                  const Parameters &prm)
        {
          out << "k1=" << prm.k1 << ", "
              << "k2=" << prm.k2 ;
          return out;
        }
    };

    // subroutine defining how the right hand side of the ODE is formed
    virtual std::valarray<double> right_hand_side(const std::valarray<double>& x,
                                                  const ParametersBase& parameters) const;



    virtual double returnConcentration(const std::valarray<double>& particleSizeDistribution,
                                       const unsigned int& particleSize,
                                       const ParametersBase& parameters) const;



    // subroutine describing how to interpret the right hand side entries
    virtual unsigned int particleSizeToIndex(const unsigned int& particleSize,
                                                const ParametersBase& parameters) const;



    virtual unsigned int getSmallestParticleSize(const ParametersBase& parameters) const;



    virtual unsigned int getLargestParticleSize(const ParametersBase& parameters) const;
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
      double k1, k2, k_forward, k_backward;
      double solvent;
      unsigned int w, maxsize, n_variables;

      // default constructor. creates an invalid object
      Parameters();

      // constructor
      Parameters(const double k_forward_value,
             const double k_backward_value,
             const double k1_value,
             const double k2_value,
             const double solvent_value,
             const unsigned int nucleation_order,
             const unsigned int maxsize_value);
             
      // Setting parameters equal, skipping the const members
      Parameters operator = (const Parameters &prm);
      
      friend
      std::ostream &
      operator<< (std::ostream &out,
                  const Parameters &prm)
        {
          out << "kb=" << prm.k_backward << ", "
              << "k1=" << prm.k1 << ", "
              << "k2=" << prm.k2;
          return out;
        }
    };

    // subroutine defining how the right hand side of the ODE is formed
    virtual std::valarray<double> right_hand_side(const std::valarray<double>& x,
                       const ParametersBase& parameters) const;



    virtual double returnConcentration(const std::valarray<double>& particleSizeDistribution,
                                       const unsigned int& particleSize,
                                       const ParametersBase& parameters) const;



    // subroutine describing how to interpret the right hand side entries
    virtual unsigned int particleSizeToIndex(const unsigned int& particleSize,
                                                const ParametersBase& parameters) const;



    virtual unsigned int getSmallestParticleSize(const ParametersBase& parameters) const;



    virtual unsigned int getLargestParticleSize(const ParametersBase& parameters) const;
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
      unsigned int w, maxsize, n_variables;
      // physically, this should be an integer, but this
      // is a variable parameter and if we wish to perform
      // statistics (e.g. compute a mean value from a 
      // probability distribution) we need this to be stored
      // as a double.
      double particle_size_cutoff;

      // default constructor. creates an invalid object
      Parameters();

      // constructor
      Parameters(const double k1_value,
             const double k2_value,
             const double k3_value,
             const unsigned int nucleation_order,
             const unsigned int maxsize_value,
             const double particle_size_cutoff_value);

      // Setting parameters equal, skipping the const members
      Parameters operator = (const Parameters &prm);
      
      friend
      std::ostream &
      operator<< (std::ostream &out,
                  const Parameters &prm)
        {
          out << "k1=" << prm.k1 << ", "
              << "k2=" << prm.k2 << ", "
              << "k3=" << prm.k3 << ", "
              << "particle_size_cutoff=" << prm.particle_size_cutoff;
          return out;
        }
    };




    // The growth rate for particles depends on their size.
    double rate_constant(const unsigned int& size,
                         const ParametersBase& parameters) const;


    // subroutine defining how the right hand side of the ODE is formed
    virtual std::valarray<double> right_hand_side(const std::valarray<double>& x,
                                                  const ParametersBase& parameters) const;



    virtual double returnConcentration(const std::valarray<double>& particleSizeDistribution,
                                       const unsigned int& particleSize,
                                       const ParametersBase& parameters) const;



    // subroutine describing how to interpret the right hand side entries
    virtual unsigned int particleSizeToIndex(const unsigned int& particleSize,
                                                const ParametersBase& parameters) const;



    virtual unsigned int getSmallestParticleSize(const ParametersBase& parameters) const;



    virtual unsigned int getLargestParticleSize(const ParametersBase& parameters) const;
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
      double k1, k2, k3, k_forward, k_backward;
      const double solvent;
      const unsigned int w, maxsize, n_variables;
      // physically, this should be an integer, but this
      // is a variable parameter and if we wish to perform
      // statistics (e.g. compute a mean value from a 
      // probability distribution) we need this to be stored
      // as a double.
      double particle_size_cutoff;

      // default constructor. creates an invalid object
      Parameters();
      
      // constructor
      Parameters(const double k_forward_value,
             const double k_backward_value,
             const double k1_value,
             const double k2_value,
             const double k3_value,
             const double solvent_value,
             const unsigned int nucleation_order,
             const unsigned int maxsize_value,
             const double particle_size_cutoff_value);

      // Setting parameters equal, skipping the const members
      Parameters operator = (const Parameters &prm);

      // Convert the current parameters object into a tuple of numbers
      // that contains all of the free variables we want to optimize
      // over.
      operator std::valarray<double> () const;
      
      friend
      std::ostream &
      operator<< (std::ostream &out,
                  const Parameters &prm)
        {
          out << "kb=" << prm.k_backward << ", "
              << "k1=" << prm.k1 << ", "
              << "k2=" << prm.k2 << ", "
              << "k3=" << prm.k3 << ", "
              << "particle_size_cutoff=" << prm.particle_size_cutoff;
          return out;
        }
    };

    // The growth rate for particles depends on their size.
    double rate_constant(const unsigned int& size,
                         const ParametersBase& parameters) const;

    // subroutine defining how the right hand side of the ODE is formed
    virtual std::valarray<double> right_hand_side(const std::valarray<double>& x,
                                                  const ParametersBase& parameters) const;



    virtual double returnConcentration(const std::valarray<double>& particleSizeDistribution,
                                       const unsigned int& particleSize,
                                       const ParametersBase& parameters) const;



    // subroutine describing how to interpret the right hand side entries
    virtual unsigned int particleSizeToIndex(const unsigned int& particleSize,
                                                const ParametersBase& parameters) const;



    virtual unsigned int getSmallestParticleSize(const ParametersBase& parameters) const;



    virtual unsigned int getLargestParticleSize(const ParametersBase& parameters) const;
  };



  // Integrate the ODE
  // class to hold integration hyperparameters
  class explEulerParameters
  {
  public:
    double startTime;
    std::vector<double> outputTimes;
    std::valarray<double> initialCondition;

    // constructor
    explEulerParameters(const double startTimeValue,
                        const std::vector<double> outputTimes,
                        const std::valarray<double> initialConditionValues);
  };


  // Explicit Euler
  std::vector<std::valarray<double>> integrate_ode_explicit_euler(const Models::explEulerParameters solverParameters,
                                                     const Models::ModelsBase& model,
                                                     const Models::ParametersBase& parameters,
                                                     const double time_step);
}


#endif /* MODELS_H_ */
