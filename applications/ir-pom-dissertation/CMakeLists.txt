CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(calc_likelihood_4step
        calc_likelihood_4step.cpp)
TARGET_LINK_LIBRARIES (calc_likelihood_4step
        libmepbm)

ADD_EXECUTABLE(calc_likelihood_3step
        calc_likelihood_3step.cpp)
TARGET_LINK_LIBRARIES (calc_likelihood_3step
        libmepbm)

ADD_EXECUTABLE(calc_likelihood_model_select
        calc_likelihood_model_select.cpp)
TARGET_LINK_LIBRARIES (calc_likelihood_model_select
        libmepbm)

ADD_EXECUTABLE(calc_concentrations_3step
        calc_concentrations_3step.cpp)
TARGET_LINK_LIBRARIES (calc_concentrations_3step
        libmepbm)

ADD_EXECUTABLE(calc_bin_probs_3step
        calc_bin_probs_3step.cpp)
TARGET_LINK_LIBRARIES (calc_bin_probs_3step
        libmepbm)

ADD_EXECUTABLE(calc_bin_probs_4step
        calc_bin_probs_4step.cpp)
TARGET_LINK_LIBRARIES (calc_bin_probs_4step
        libmepbm)

ADD_EXECUTABLE(calc_likelihood_3step_simplified_constants
        calc_likelihood_3step_simplified_constants.cpp)
TARGET_LINK_LIBRARIES (calc_likelihood_3step_simplified_constants
        libmepbm)


ADD_EXECUTABLE(calc_concentrations_3step_simple
        calc_concentrations_3step_simple.cpp)
TARGET_LINK_LIBRARIES (calc_concentrations_3step_simple
        libmepbm)