CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(process_samples_3step
        process_samples_3step.cpp)
TARGET_LINK_LIBRARIES (process_samples_3step
        libmepbm)

ADD_EXECUTABLE(process_samples_4step
        process_samples_4step.cpp)
TARGET_LINK_LIBRARIES (process_samples_4step
        libmepbm)