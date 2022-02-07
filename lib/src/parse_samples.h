#ifndef MEPBM_SAMPLE_PARSER_H
#define MEPBM_SAMPLE_PARSER_H

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <iomanip>



namespace MEPBM {
  template<typename Real>
  std::pair<std::vector<std::vector<double>>,std::vector<double>>
  parse_samples(const std::string & file_name, const unsigned int n_parameters)
  {
    std::vector< std::vector<double> > samples;
    std::vector<double> likelihood_values;

    std::ifstream infile(file_name);

    std::string line;
    while (std::getline(infile, line))
    {
      /*
       * There are two lines we care about:
       *    Sample: ...
       *    and
       *       relative log likelihood -> ...
       */
      static const std::string sample_key = "Sample:";
      static const std::string likelihood_key = "   relative log likelihood ->";
      if (line.substr(0,sample_key.length()) == sample_key)
      {
        const unsigned int pos = sample_key.length()+1; // +1 because there is a space between the keyword and what we want
        const unsigned int str_length = line.size();
        const unsigned int n = str_length - pos;
        // Starting after the keyword, add parameters to a vector
        std::vector<double> parameters(n_parameters);
        unsigned int p = 0;
        std::istringstream sample_stream(line.substr(pos, n));
        while (sample_stream)
        {
          sample_stream >> parameters[p];
          ++p;
        }
        samples.push_back(parameters);
      }
      else if (line.substr(0,likelihood_key.length()) == likelihood_key)
      {
        const unsigned int pos = likelihood_key.length() + 1;
        const unsigned int str_length = line.size();
        const unsigned int n = str_length - pos;
        // Starting after the keyword, add the likelihood to the ordered list of likelihood values
        double ll;
        std::istringstream likelihood_stream(line.substr(pos, n));
        likelihood_stream >> ll;
        likelihood_values.push_back(ll);
      }
    }
    return {samples, likelihood_values};
  }
}

#endif //MEPBM_SAMPLE_PARSER_H
