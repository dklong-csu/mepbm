CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(generate_samples_3step_all
               mcmc.cpp
               $<TARGET_OBJECTS:libmepbm>)
TARGET_LINK_LIBRARIES (generate_samples_3step_all
                       Threads::Threads)