CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_SUBDIRECTORY(mech1_A_3step)

ADD_SUBDIRECTORY(mech1_A_3step_noprecursor)

ADD_SUBDIRECTORY(mech1_A_4step)

ADD_SUBDIRECTORY(mech1_A_4step_noprecursor)

ADD_EXECUTABLE(process_samples_3step
        process_samples_3step.cpp)
TARGET_LINK_LIBRARIES (process_samples_3step
        libmepbm)

ADD_EXECUTABLE(process_samples_4step
        process_samples_4step.cpp)
TARGET_LINK_LIBRARIES (process_samples_4step
        libmepbm)

ADD_EXECUTABLE(ode_solution_mech1_A_3step
        ode_solution_mech1_A_3step.cpp)
TARGET_LINK_LIBRARIES (ode_solution_mech1_A_3step
        libmepbm)

ADD_EXECUTABLE(ode_solution_mech1_A_4step
        ode_solution_mech1_A_4step.cpp)
TARGET_LINK_LIBRARIES (ode_solution_mech1_A_4step
        libmepbm)