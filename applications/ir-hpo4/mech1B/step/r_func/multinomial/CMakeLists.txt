CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(calc_log_likelihood_1B_step_r_multi
        calc_log_likelihood.cpp)
TARGET_LINK_LIBRARIES(calc_log_likelihood_1B_step_r_multi
        libmepbm)

ADD_EXECUTABLE(solve_ode_1B_step_r_multi
        solve_ode.cpp)
TARGET_LINK_LIBRARIES(solve_ode_1B_step_r_multi
        libmepbm)