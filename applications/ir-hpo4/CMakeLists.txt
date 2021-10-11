CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)


ADD_EXECUTABLE(mcmc_preliminary
        mcmc_preliminary.cpp)
TARGET_LINK_LIBRARIES(mcmc_preliminary
        libmepbm)
