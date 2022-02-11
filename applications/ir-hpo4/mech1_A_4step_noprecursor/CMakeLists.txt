CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(mcmc_mech1_A_4step_noprecursor
        mcmc_mech1_A_4step_noprecursor.cpp)
TARGET_LINK_LIBRARIES (mcmc_mech1_A_4step_noprecursor
        libmepbm)

ADD_EXECUTABLE(demc_mech1_A_4step_noprecursor
        demc_mech1_A_4step_noprecursor.cpp)
TARGET_LINK_LIBRARIES (demc_mech1_A_4step_noprecursor
        libmepbm)