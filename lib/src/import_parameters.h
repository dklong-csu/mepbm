#ifndef MEPBM_IMPORT_PARAMETERS_H
#define MEPBM_IMPORT_PARAMETERS_H



#include <string>
#include <fstream>
#include <sstream>



namespace MEPBM {
  /**
   * A function that imports parameters from a file and returns an object containing those parameters.
   * @tparam SampleType - An object containing parameters accessible via the `[]` operator.
   * @param file_path  - The name of the file that contains parameters formatted in a single column.
   * @param n_prm - The number of parameters expected.
   * @return
   */
  template<typename SampleType>
  SampleType
  import_parameters(const std::string file_path, const unsigned int n_prm)
  {
    std::string line;
    std::ifstream input_file(file_path);
    SampleType prm(n_prm);
    if (input_file.is_open())
    {
      unsigned int index = 0;
      while ( getline(input_file,line) )
      {
        std::istringstream ss_in(line);
        double prm_val = 0.0;
        ss_in >> prm_val;
        prm[index] = prm_val;
        ++index;
      }
      input_file.close();
      // if index < n_prm the not enough parameters were passed
      if (index < n_prm) {
        std::string err_msg = "Fewer than "
                              + std::to_string(n_prm)
                              + " were passed. Please update "
                              + file_path
                              + " to include "
                              + std::to_string(n_prm)
                              + " values arranged in a single column.\n";
        throw std::invalid_argument(err_msg);
      }
    }
    else {
      throw std::runtime_error("Unable to open file\n");
    }

    return prm;
  }
}

#endif //MEPBM_IMPORT_PARAMETERS_H
