CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(calc_cost_function
        calc_cost_function.cpp)
TARGET_LINK_LIBRARIES (calc_cost_function
        libmepbm)

ADD_EXECUTABLE(convergence_check
        convergence_check.cpp)
TARGET_LINK_LIBRARIES (convergence_check
        libmepbm)


ADD_EXECUTABLE(calc_size_distribution
        calc_size_distribution.cpp)
TARGET_LINK_LIBRARIES(calc_size_distribution
        libmepbm)

ADD_EXECUTABLE(calc_size_distribution_LFIRE
        calc_size_distribution_LFIRE.cpp)
TARGET_LINK_LIBRARIES(calc_size_distribution_LFIRE
        libmepbm)