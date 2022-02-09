CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(mcmc_mech1_A_4step
        mcmc_mech1_A_4step.cpp)
TARGET_LINK_LIBRARIES (mcmc_mech1_A_4step
        libmepbm)