#ifndef MEPBM_CHECK_SUNDIALS_FLAGS_H
#define MEPBM_CHECK_SUNDIALS_FLAGS_H



#include <iostream>
#include <string>



namespace MEPBM {
  /// Enumeration to make checking of SUNDIALS C functions easier
  enum SuccessDefinition {MEMORY, RETURNZERO, RETURNNONNEGATIVE};



  /// Function to check if the C-style SUNDIALS functions are successful
  int check_flag(void *flag_value, const std::string &function_name, SuccessDefinition success_type)
  {
    int result = 0;

    // For checking when success is defined by a return value.
    int error_flag;

    switch (success_type) {
      case MEMORY :
        if (flag_value == nullptr)
        {
          std::cerr << std::endl
                    << "ERROR: "
                    << function_name
                    << " returned NULL pointer."
                    << std::endl;
          // Return 1 to indicate failure
          result = 1;
        }
        break;

      case RETURNZERO :
        error_flag = *((int *) flag_value);
        if (error_flag != 0)
        {
          std::cerr << std::endl
                    << "ERROR: "
                    << function_name
                    << " returned with flag = "
                    << error_flag
                    << std::endl;
          // Return 1 to indicate failure
          result = 1;
        }
        break;

      case RETURNNONNEGATIVE :
        error_flag = *((int *) flag_value);
        if (error_flag < 0)
        {
          std::cerr << std::endl
                    << "ERROR: "
                    << function_name
                    << " returned with flag = "
                    << error_flag
                    << std::endl;
          // Return 1 to indicate failure
          result = 1;
        }
        break;

      default:
        std::cerr << std::endl
                  << "ERROR: check_flag called with an invalid SuccessDefinition."
                  << std::endl;
        // Return 1 to indicate failure
        result = 1;
        break;
    }

    return result;
  }
}

#endif //MEPBM_CHECK_SUNDIALS_FLAGS_H
