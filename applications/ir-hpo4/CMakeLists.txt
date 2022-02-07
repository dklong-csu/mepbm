CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)


ADD_EXECUTABLE(mcmc_mech1_A_3step
               mcmc_mech1_A_3step.cpp)
TARGET_LINK_LIBRARIES (mcmc_mech1_A_3step
                       libmepbm)

ADD_EXECUTABLE(process_samples
        process_samples.cpp)
TARGET_LINK_LIBRARIES (process_samples
        libmepbm)