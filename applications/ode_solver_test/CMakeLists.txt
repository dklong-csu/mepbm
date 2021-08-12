CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)


ADD_EXECUTABLE(ode_solver_test_3step
        ode_solver_test_3step.cpp)
TARGET_LINK_LIBRARIES(ode_solver_test_3step
        libmepbm)

ADD_EXECUTABLE(ode_solver_test_4step
        ode_solver_test_4step.cpp)
TARGET_LINK_LIBRARIES(ode_solver_test_4step
        libmepbm)