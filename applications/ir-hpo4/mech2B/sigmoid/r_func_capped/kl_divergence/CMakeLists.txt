CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(calc_log_likelihood_2B_sig_rcap_kl
        calc_log_likelihood.cpp)
TARGET_LINK_LIBRARIES(calc_log_likelihood_2B_sig_rcap_kl
        libmepbm)

ADD_EXECUTABLE(solve_ode_2B_sig_rcap_kl
        solve_ode.cpp)
TARGET_LINK_LIBRARIES(solve_ode_2B_sig_rcap_kl
        libmepbm)