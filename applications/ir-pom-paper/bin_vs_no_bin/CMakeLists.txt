CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(random_samples_bin_compare
               random_samples.cpp)
TARGET_LINK_LIBRARIES (random_samples_bin_compare
                       libmepbm)
