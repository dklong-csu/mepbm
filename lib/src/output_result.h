#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>


namespace MEPBM {
  template<typename ResultType>
  void
  output_result(const ResultType & result, const std::string file_path) {
    std::ofstream outputfile(file_path.c_str());
    outputfile << std::setprecision(40) << result << std::endl;
    outputfile.close();
  }
}