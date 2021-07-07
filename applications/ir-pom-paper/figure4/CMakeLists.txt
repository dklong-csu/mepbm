CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(plot_random_samples
               plot_random_samples.cpp)
TARGET_LINK_LIBRARIES (plot_random_samples
                       libmepbm)
