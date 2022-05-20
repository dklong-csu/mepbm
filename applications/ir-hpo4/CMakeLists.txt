CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(calc_log_likelihood
        calc_log_likelihood.cpp)
TARGET_LINK_LIBRARIES(calc_log_likelihood
        libmepbm)

ADD_EXECUTABLE(solve_ode
        solve_ode.cpp)
TARGET_LINK_LIBRARIES(solve_ode
        libmepbm)