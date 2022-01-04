CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(summarize_samples
        summarize_samples.cpp)
TARGET_LINK_LIBRARIES(summarize_samples
        libmepbm)

ADD_EXECUTABLE(ode_solver_timing_test
        ode_solver_timing_test.cpp)
TARGET_LINK_LIBRARIES(ode_solver_timing_test
        libmepbm)

ADD_EXECUTABLE(mcmc_adaptive
        mcmc_adaptive.cpp)
TARGET_LINK_LIBRARIES(mcmc_adaptive
        libmepbm)

ADD_EXECUTABLE(mcmc_demc
        mcmc_demc.cpp)
TARGET_LINK_LIBRARIES(mcmc_demc
        libmepbm)

ADD_EXECUTABLE(solve_single_ode
        solve_single_ode.cpp)
TARGET_LINK_LIBRARIES(solve_single_ode
        libmepbm)