CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

MESSAGE(STATUS "Setting up Ir-POM CVODE Test")

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(mcmc_uniform
        mcmc_uniform.cpp)
TARGET_LINK_LIBRARIES (mcmc_uniform
        libmepbm)

ADD_EXECUTABLE(mean_value_solution
        mean_value_solution.cpp)
TARGET_LINK_LIBRARIES (mean_value_solution
        libmepbm)