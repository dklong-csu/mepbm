#include <cmath>
#include <valarray>
#include <stdexcept>
#include "models.h"



// Parameters constructors
Models::TwoStep::Parameters::Parameters()
:
// just defer to the other constructor, using invalid values
Parameters(std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<unsigned int>::max(),
           std::numeric_limits<unsigned int>::max())
{}



Models::TwoStep::Parameters::Parameters(const double k1_value,
                                        const double k2_value,
                                        const unsigned int nucleation_order,
                                        const unsigned int maxsize_value)
:
  k1 (k1_value),
  k2 (k2_value),
  w (nucleation_order),
  maxsize (maxsize_value),
  n_variables (maxsize - w + 2),
{}



Models::TwoStepAlternative::Parameters::Parameters()
:
// just defer to the other constructor, using invalid values
Parameters(std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<unsigned int>::max(),
           std::numeric_limits<unsigned int>::max())
{}


Models::TwoStepAlternative::Parameters::Parameters(const double k_forward_value,
                                                   const double k_backward_value,
                                                   const double k1_value,
                                                   const double k2_value,
                                                   const double solvent_value,
                                                   const unsigned int nucleation_order,
                                                   const unsigned int maxsize_value)
:
  k_forward (k_forward_value),
  k_backward (k_backward_value),
  k1 (k1_value),
  k2 (k2_value),
  w (nucleation_order),
  maxsize (maxsize_value),
  n_variables (maxsize - w + 4),
  solvent (solvent_value),
{}



Models::ThreeStep::Parameters::Parameters()
:
// just defer to the other constructor, using invalid values
Parameters(std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<unsigned int>::max(),
           std::numeric_limits<unsigned int>::max(),
           std::numeric_limits<unsigned int>::max())
{}



Models::ThreeStep::Parameters::Parameters(const double k1_value,
                                          const double k2_value,
                                          const double k3_value,
                                          const unsigned int nucleation_order,
                                          const unsigned int maxsize_value,
                                          const unsigned int particle_size_cutoff_value)
:
  k1 (k1_value),
  k2 (k2_value),
  k3 (k3_value),
  w (nucleation_order),
  maxsize (maxsize_value),
  n_variables (maxsize - w + 2),
  particle_size_cutoff (particle_size_cutoff_value)
{}



Models::ThreeStepAlternative::Parameters::Parameters()
:
// just defer to the other constructor, using invalid values
Parameters(std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<double>::signaling_NaN(),
           std::numeric_limits<unsigned int>::max(),
           std::numeric_limits<unsigned int>::max(),
           std::numeric_limits<unsigned int>::max())
{}

           

Models::ThreeStepAlternative::Parameters::Parameters(const double k_forward_value,
                                                     const double k_backward_value,
                                                     const double k1_value,
                                                     const double k2_value,
                                                     const double k3_value,
                                                     const double solvent_value,
                                                     const unsigned int nucleation_order,
                                                     const unsigned int maxsize_value,
                                                     const unsigned int particle_size_cutoff_value)
:
k_forward (k_forward_value),
  k_backward (k_backward_value),
  k1 (k1_value),
  k2 (k2_value),
  k3 (k3_value),
  w (nucleation_order),
  maxsize (maxsize_value),
  n_variables (maxsize - w + 4),
  particle_size_cutoff (particle_size_cutoff_value),
  solvent (solvent_value)
{}



Models::ThreeStepAlternative::Parameters
Models::ThreeStepAlternative::Parameters::operator += (const Parameters &prm)
{
  k1 += prm.k1;
  k2 += prm.k2;
  k3 += prm.k3;
  k_forward += prm.k_forward;
  k_backward += prm.k_backward;
  solvent += prm.solvent;
  w += prm.w;
  maxsize += prm.maxsize;
  n_variables += prm.n_variables;
  particle_size_cutoff += prm.particle_size_cutoff;

  return *this;
}



Models::ThreeStepAlternative::Parameters
Models::ThreeStepAlternative::Parameters::operator -= (const Parameters &prm)
{
  k1 -= prm.k1;
  k2 -= prm.k2;
  k3 -= prm.k3;
  k_forward -= prm.k_forward;
  k_backward -= prm.k_backward;
  solvent -= prm.solvent;
  w -= prm.w;
  maxsize -= prm.maxsize;
  n_variables -= prm.n_variables;
  particle_size_cutoff -= prm.particle_size_cutoff;

  return *this;
}


Models::ThreeStepAlternative::Parameters
Models::ThreeStepAlternative::Parameters::operator /= (const unsigned int n)
{
  k1 /= n;
  k2 /= n;
  k3 /= n;
  k_forward /= n;
  k_backward /= n;
  solvent /= n;
  w /= n;
  maxsize /= n;
  n_variables /= n;
  particle_size_cutoff /= n;

  return *this;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// helper functions

double Models::available_atoms(const double& size)
{
  return size * 2.677 * std::pow(size, -0.28);
}



double Models::ThreeStep::rate_constant(const unsigned int& size,
                                        const Models::ParametersBase& parameters) const
{
  const Models::ThreeStep::Parameters& three_step_parameters = dynamic_cast<const Models::ThreeStep::Parameters&>(parameters);
  if (size <= three_step_parameters.particle_size_cutoff)
    return three_step_parameters.k2;
  else
    return three_step_parameters.k3;
}



double Models::ThreeStepAlternative::rate_constant(const unsigned int& size,
                                                   const Models::ParametersBase& parameters) const
{
  const Models::ThreeStepAlternative::Parameters& three_step_alt_parameters = dynamic_cast<const Models::ThreeStepAlternative::Parameters&>(parameters);
  if (size <= three_step_alt_parameters.particle_size_cutoff)
    return three_step_alt_parameters.k2;
  else
    return three_step_alt_parameters.k3;
}



// convert particle size to index of the solution/right hand side vector

unsigned int Models::TwoStep::particleSizeToIndex(const unsigned int& particleSize,
                                                  const ParametersBase& parameters) const
{
  const Models::TwoStep::Parameters& twoStepParameters = dynamic_cast<const Models::TwoStep::Parameters&>(parameters);

  if (particleSize == 1)
    return 0;
  else if ((particleSize >= twoStepParameters.w) && (particleSize <= twoStepParameters.maxsize))
      return particleSize - twoStepParameters.w + 1;
  else
    throw std::domain_error("Particle size outside of tracked range. Check the nucleation order and maximum particle size.");
}



unsigned int Models::TwoStepAlternative::particleSizeToIndex(const unsigned int& particleSize,
                                                             const ParametersBase& parameters) const
{
  const Models::TwoStepAlternative::Parameters& twoStepAltParameters = dynamic_cast<const Models::TwoStepAlternative::Parameters&>(parameters);

  if (particleSize == 1)
    return 0;
  else if ((particleSize >= twoStepAltParameters.w) && (particleSize <= twoStepAltParameters.maxsize))
      return particleSize - twoStepAltParameters.w + 3;
  else
    throw std::domain_error("Particle size outside of tracked range. Check the nucleation order and maximum particle size.");
}



unsigned int Models::ThreeStep::particleSizeToIndex(const unsigned int& particleSize,
                                                    const ParametersBase& parameters) const
{
  const Models::ThreeStep::Parameters& threeStepParameters = dynamic_cast<const Models::ThreeStep::Parameters&>(parameters);

  if (particleSize == 1)
    return 0;
  else if ((particleSize >= threeStepParameters.w) && (particleSize <= threeStepParameters.maxsize))
      return particleSize - threeStepParameters.w + 1;
  else
    throw std::domain_error("Particle size outside of tracked range. Check the nucleation order and maximum particle size.");
}



unsigned int Models::ThreeStepAlternative::particleSizeToIndex(const unsigned int& particleSize,
                                                               const ParametersBase& parameters) const
{
  const Models::ThreeStepAlternative::Parameters& threeStepAltParameters = dynamic_cast<const Models::ThreeStepAlternative::Parameters&>(parameters);

  if (particleSize == 1)
    return 0;
  else if ((particleSize >= threeStepAltParameters.w) && (particleSize <= threeStepAltParameters.maxsize))
      return particleSize - threeStepAltParameters.w + 3;
  else
    throw std::domain_error("Particle size outside of tracked range. Check the nucleation order and maximum particle size.");
}



// get largest and smallest recognized particle sizes
unsigned int Models::TwoStep::getSmallestParticleSize(const ParametersBase& parameters) const
{
  const Models::TwoStep::Parameters& twoStepParameters = dynamic_cast<const Models::TwoStep::Parameters&>(parameters);
  return twoStepParameters.w;
}



unsigned int Models::TwoStep::getLargestParticleSize(const ParametersBase& parameters) const
{
  const Models::TwoStep::Parameters& twoStepParameters = dynamic_cast<const Models::TwoStep::Parameters&>(parameters);
  return twoStepParameters.maxsize;
}



unsigned int Models::TwoStepAlternative::getSmallestParticleSize(const ParametersBase& parameters) const
{
  const Models::TwoStepAlternative::Parameters& twoStepAltParameters = dynamic_cast<const Models::TwoStepAlternative::Parameters&>(parameters);
  return twoStepAltParameters.w;
}



unsigned int Models::TwoStepAlternative::getLargestParticleSize(const ParametersBase& parameters) const
{
  const Models::TwoStepAlternative::Parameters& twoStepAltParameters = dynamic_cast<const Models::TwoStepAlternative::Parameters&>(parameters);
  return twoStepAltParameters.maxsize;
}



unsigned int Models::ThreeStep::getSmallestParticleSize(const ParametersBase& parameters) const
{
  const Models::ThreeStep::Parameters& threeStepParameters = dynamic_cast<const Models::ThreeStep::Parameters&>(parameters);
  return threeStepParameters.w;
}



unsigned int Models::ThreeStep::getLargestParticleSize(const ParametersBase& parameters) const
{
  const Models::ThreeStep::Parameters& threeStepParameters = dynamic_cast<const Models::ThreeStep::Parameters&>(parameters);
  return threeStepParameters.maxsize;
}



unsigned int Models::ThreeStepAlternative::getSmallestParticleSize(const ParametersBase& parameters) const
{
  const Models::ThreeStepAlternative::Parameters& threeStepAltParameters = dynamic_cast<const Models::ThreeStepAlternative::Parameters&>(parameters);
  return threeStepAltParameters.w;
}



unsigned int Models::ThreeStepAlternative::getLargestParticleSize(const ParametersBase& parameters) const
{
  const Models::ThreeStepAlternative::Parameters& threeStepAltParameters = dynamic_cast<const Models::ThreeStepAlternative::Parameters&>(parameters);
  return threeStepAltParameters.maxsize;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Functions to return concentrations from a solution vector of a given model.

double Models::TwoStep::returnConcentration(const std::valarray<double>& particleSizeDistribution,
                                            const unsigned int& particleSize,
                                            const ParametersBase& parameters) const
{
  const Models::TwoStep::Parameters& twoStepParameters = dynamic_cast<const Models::TwoStep::Parameters&>(parameters);

  return particleSizeDistribution[Models::TwoStep::particleSizeToIndex(particleSize, twoStepParameters)];
}



double Models::TwoStepAlternative::returnConcentration(const std::valarray<double>& particleSizeDistribution,
                                                       const unsigned int& particleSize,
                                                       const ParametersBase& parameters) const
{
  const Models::TwoStepAlternative::Parameters& twoStepAltParameters = dynamic_cast<const Models::TwoStepAlternative::Parameters&>(parameters);

  return particleSizeDistribution[Models::TwoStepAlternative::particleSizeToIndex(particleSize, twoStepAltParameters)];
}



double Models::ThreeStep::returnConcentration(const std::valarray<double>& particleSizeDistribution,
                                              const unsigned int& particleSize,
                                              const ParametersBase& parameters) const
{
  const Models::ThreeStep::Parameters& threeStepParameters = dynamic_cast<const Models::ThreeStep::Parameters&>(parameters);

  return particleSizeDistribution[Models::ThreeStep::particleSizeToIndex(particleSize, threeStepParameters)];
}



double Models::ThreeStepAlternative::returnConcentration(const std::valarray<double>& particleSizeDistribution,
                                                       const unsigned int& particleSize,
                                                       const ParametersBase& parameters) const
{
  const Models::ThreeStepAlternative::Parameters& threeStepAltParameters = dynamic_cast<const Models::ThreeStepAlternative::Parameters&>(parameters);

  return particleSizeDistribution[Models::ThreeStepAlternative::particleSizeToIndex(particleSize, threeStepAltParameters)];
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// right hand side implementations
std::valarray<double> Models::TwoStep::right_hand_side(const std::valarray<double>& x,
                                                       const Models::ParametersBase& parameters) const
{
  const Models::TwoStep::Parameters& two_step_parameters = dynamic_cast<const Models::TwoStep::Parameters&>(parameters);

  /*
  f[0] = dn_0
  f[1] = dn_w
  f[2] = dn_{w+1}
  ...
  f[two_step_parameters.n_variable - 1) = dn_{max size}
  */
  std::valarray<double> f(two_step_parameters.n_variables);
  // loss from nucleation
  f[0] = -((int)two_step_parameters.w * (int)two_step_parameters.k1 * std::pow(x[0], 1. * (int)two_step_parameters.w));
  // gain from nucleation
  f[1] = two_step_parameters.k1 * std::pow(x[0], 1. * two_step_parameters.w);
  for (unsigned int i = 2; i < two_step_parameters.n_variables; i++)
  {
    // gain from growth
    f[i] = two_step_parameters.k2 * x[0] * Models::available_atoms(two_step_parameters.w + i - 2) * x[i - 1];
    f[0] -= f[i];
  }

  for (unsigned int i = 1; i < two_step_parameters.n_variables; i++)
  {
    // loss from growth
    f[i] -= f[i + 1];
  }

  // loss from growth on largest particle
  f[two_step_parameters.n_variables - 1] -= two_step_parameters.k2 * x[0] * Models::available_atoms(two_step_parameters.maxsize) * x[two_step_parameters.n_variables - 1];
  f[0] -= two_step_parameters.k2 * x[0] * Models::available_atoms(two_step_parameters.maxsize) * x[two_step_parameters.n_variables - 1];
  return f;
}



std::valarray<double> Models::TwoStepAlternative::right_hand_side(const std::valarray<double>& x,
                                                                  const Models::ParametersBase& parameters) const
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
    f[i] = two_step_alt_parameters.k2 * x[0] * Models::available_atoms(i - 1) * x[i - 1];
    // precursor loss from growth
    f[0] -= f[i];
  }

  for (unsigned int i = 3; i < two_step_alt_parameters.n_variables - 1; i++)
  {
    // loss from growth
    f[i] -= f[i + 1];
  }

  // loss from growth on largest particle -- I'm torn about including this
  f[two_step_alt_parameters.n_variables - 1] -= two_step_alt_parameters.k2 * x[0] * Models::available_atoms(two_step_alt_parameters.maxsize) * x[two_step_alt_parameters.n_variables - 1];
  f[0] -= two_step_alt_parameters.k2 * x[0] * Models::available_atoms(two_step_alt_parameters.maxsize) * x[two_step_alt_parameters.n_variables - 1];
  // assign ligand rate now that precursor rate is final
  f[2] = -f[0];

  return f;
}



std::valarray<double> Models::ThreeStep::right_hand_side(const std::valarray<double>& x,
                                                         const Models::ParametersBase& parameters) const
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
    f[i] = Models::ThreeStep::rate_constant(three_step_parameters.w + i - 2,three_step_parameters)*x[0]* Models::available_atoms(three_step_parameters.w + i - 2) * x[i - 1];
    f[0] -= f[i];
  }

  for (unsigned int i = 1; i < three_step_parameters.n_variables - 1; i++)
  {
    // particle loss from growth
    f[i] -= f[i + 1];
  }

  // loss from growth on largest particle
  f[three_step_parameters.n_variables - 1] -= Models::ThreeStep::rate_constant(three_step_parameters.maxsize,three_step_parameters)* x[0] * Models::available_atoms(three_step_parameters.maxsize) * x[three_step_parameters.n_variables - 1];
  f[0] -= Models::ThreeStep::rate_constant(three_step_parameters.maxsize, three_step_parameters) * x[0] * Models::available_atoms(three_step_parameters.maxsize) * x[three_step_parameters.n_variables - 1];
  return f;
}



std::valarray<double> Models::ThreeStepAlternative::right_hand_side(const std::valarray<double>& x,
                                                                    const Models::ParametersBase& parameters) const
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
    f[i] = rate_constant(three_step_alt_parameters.w + i - 4, three_step_alt_parameters) * x[0] * Models::available_atoms(three_step_alt_parameters.w + i - 4) * x[i - 1];
    f[0] -= f[i];
  }

  for (unsigned int i = 3; i < three_step_alt_parameters.n_variables - 1; i++)
  {
    // loss from growth
    f[i] -= f[i + 1];
  }

  // loss from growth on largest particle -- I'm torn about including this
  f[three_step_alt_parameters.n_variables - 1] -= Models::ThreeStepAlternative::rate_constant(three_step_alt_parameters.maxsize,three_step_alt_parameters) * x[0] * Models::available_atoms(three_step_alt_parameters.maxsize) * x[three_step_alt_parameters.n_variables - 1];
  f[0] -= Models::ThreeStepAlternative::rate_constant(three_step_alt_parameters.maxsize, three_step_alt_parameters) * x[0] * Models::available_atoms(three_step_alt_parameters.maxsize) * x[three_step_alt_parameters.n_variables - 1];
  // assign ligand rate now that precursor rate is final
  f[2] = -f[0];

  return f;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// class to hold integration hyperparameters
Models::explEulerParameters::explEulerParameters(const double startTimeValue,
                      const double endTimeValue,
                      const std::valarray<double> initialConditionValues)
{
  startTime = startTimeValue;
  endTime = endTimeValue;
  initialCondition = initialConditionValues;
}



// ODE solver
std::valarray<double> Models::integrate_ode_explicit_euler(const Models::explEulerParameters solverParameters,
                                                   const Models::ModelsBase& model,
                                                   const Models::ParametersBase& modelParameters)
{
  std::valarray<double> x = solverParameters.initialCondition;
  double time_step = 1e-5;

  double time = solverParameters.startTime;
  while (time < solverParameters.endTime)
  {
    // advance to next time
    if (time + time_step > solverParameters.endTime)
    {
      time_step = solverParameters.endTime - time;
      time = solverParameters.endTime;
    }
    else
    {
      time += time_step;
    }

    // explicit euler update step
    x += time_step * model.right_hand_side(x, modelParameters);
  }

  return x;
}
