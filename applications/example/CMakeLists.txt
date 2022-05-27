CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE(Threads)

# Change the executable name to something meaningful
ADD_EXECUTABLE(calc_log_likelihood_example
    calc_log_likelihood.cpp)
TARGET_LINK_LIBRARIES(calc_log_likelihood_example
    libmepbm)

# Change the executable name to something meaningful
ADD_EXECUTABLE(solve_ode_example
    solve_ode.cpp)
TARGET_LINK_LIBRARIES(solve_ode_example
    libmepbm)