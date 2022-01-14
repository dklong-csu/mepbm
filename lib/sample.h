#ifndef MEPBM_SAMPLE_H
#define MEPBM_SAMPLE_H

#include "nvector_eigen.h"
#include "sunmatrix_eigen.h"
#include "src/models.h"
#include "sampling_parameters.h"
#include <utility>
#include <vector>
#include "src/histogram.h"
#include <functional>
#include <utility>
#include <stdexcept>
#include <valarray>
#include <limits>



namespace Sampling
{
  /**
   * A Sample is a data structure that describes the adjustable parameters in a model as well as the necessary
   * information to compute a likelihood of the sample.
   */
  template<typename RealType>
  class Sample
  {
  public:
    /// Default constructor
    Sample() = default;

    /// A constructor taking in the necessary arguments to fully define a sample in the desired parameter space.
    Sample(std::vector<RealType> real_prm,
           std::vector<int> int_prm);

    /// Copy constructor
    Sample(const Sample& old_prm);

    /// Copy assignment
    Sample & operator=(const Sample& old_prm);

    /// Destructor
    ~Sample() = default;

    /// Returns the number of parameters
    int
    get_dimension() const;

    /// The values of all the real-valued parameters.
    std::vector<RealType> real_valued_parameters;

    /// The values of all the integer-valued parameters.
    std::vector<int> integer_valued_parameters;

    /// Conversion to std::valarray<RealType> for arithmetic purposes
    explicit operator std::valarray<RealType> () const;

    /// What it means to output a Sample
    template<typename R>
    friend std::ostream & operator<< (std::ostream &out, const Sample<R> &sample);
  };



  template<typename RealType>
  Sample<RealType>::Sample(std::vector<RealType> real_prm,
                           std::vector<int> int_prm)
   : real_valued_parameters(real_prm),
     integer_valued_parameters(std::move(int_prm))
  {}



  template<typename RealType>
  Sample<RealType>::Sample(const Sample & old_prm)
  {
    real_valued_parameters = old_prm.real_valued_parameters;
    integer_valued_parameters = old_prm.integer_valued_parameters;
  }



  template<typename RealType>
  Sample<RealType> &
  Sample<RealType>::operator=(const Sample & old_prm)
  {
    real_valued_parameters = old_prm.real_valued_parameters;
    integer_valued_parameters = old_prm.integer_valued_parameters;
    return *this;
  }



  template<typename RealType>
  int
  Sample<RealType>::get_dimension() const
  {
    return real_valued_parameters.size() + integer_valued_parameters.size();
  }



  template<typename RealType>
  Sample<RealType>::operator std::valarray<RealType>() const
  {
    const auto dim = real_valued_parameters.size() + integer_valued_parameters.size();
    std::valarray<RealType> vec(dim);

    for (unsigned int i=0; i<real_valued_parameters.size(); ++i)
    {
      vec[i] = real_valued_parameters[i];
    }

    for (unsigned int i=0; i<integer_valued_parameters.size(); ++i)
    {
      vec[i+real_valued_parameters.size()] = static_cast<RealType>(integer_valued_parameters[i]);
    }
    return vec;
  }



  template<typename RealType>
  std::ostream &operator<<(std::ostream &out, const Sample<RealType> &sample)
  {
    const auto dim_real = sample.real_valued_parameters.size();
    const auto dim_int = sample.integer_valued_parameters.size();

    for (unsigned int i=0; i<dim_real; ++i)
    {
      if (i == dim_real - 1 && dim_int == 0)
      {
        out << sample.real_valued_parameters[i];
      }
      else
      {
        out << sample.real_valued_parameters[i] << ", ";
      }
    }

    for (unsigned int i=0; i<dim_int; ++i)
    {
      if (i == dim_int - 1)
      {
        out << sample.integer_valued_parameters[i];
      }
      else
      {
        out << sample.integer_valued_parameters[i] << ", ";
      }
    }

    return out;
  }



  /**
   * A function that checks if a sample is valid based on whether the parameters are within the domain bounds
   * specified by the user.
   */
  template<typename RealType>
  bool
  sample_is_valid(const Sample<RealType> &sample,
                  const std::vector<std::pair<RealType,RealType>> &real_domain,
                  const std::vector<std::pair<int,int>> &int_domain)
  {
    if (sample.real_valued_parameters.size() != real_domain.size())
      throw std::domain_error("The number of real-valued parameters being checked is not equal to the number of parameter bounds.");

    if (sample.integer_valued_parameters.size() != int_domain.size())
      throw std::domain_error("The number of integer-valued parameters being checked is not equal to the number of parameter bounds.");

    for (unsigned int i=0; i<sample.real_valued_parameters.size(); ++i)
    {
      if (sample.real_valued_parameters[i] < real_domain[i].first
          || sample.real_valued_parameters[i] > real_domain[i].second)
        return false;
    }

    for (unsigned int i=0; i<sample.integer_valued_parameters.size(); ++i)
    {
      if (sample.integer_valued_parameters[i] < int_domain[i].first
          || sample.integer_valued_parameters[i] > int_domain[i].second)
        return false;
    }

    return true;
  }
}

#endif //MEPBM_SAMPLE_H
