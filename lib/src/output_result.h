#ifndef MEPBM_OUTPUT_RESULT_H
#define MEPBM_OUTPUT_RESULT_H



#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>


namespace MEPBM {
  /**
   * A function that exports a result to a specified file.
   * @tparam ResultType - Type associated with the result being exported. ResultType must have a `std::ofstream <<` operator defined.
   * @param result - The information to be output to a file.
   * @param file_path - The name of the file containing the output.
   */
  template<typename ResultType>
  void
  output_result(const ResultType & result, const std::string file_path) {
    std::ofstream outputfile(file_path.c_str());
    outputfile << std::setprecision(40) << result << std::endl;
    outputfile.close();
  }
}

#endif // MEPBM_OUTPUT_RESULT_H