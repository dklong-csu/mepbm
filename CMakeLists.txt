CMAKE_MINIMUM_REQUIRED (VERSION 3.1)
PROJECT (mepbm CXX)

#########################################
### Set up compiler flags and input paths

# Compile everything with C++11
SET (CMAKE_CXX_STANDARD 11)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)
SET (CMAKE_CXX_EXTENSIONS OFF)

# Find include files in the current directory
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})


#########################################
# Describe what files should go into the library that contains all of
# the functionality that is used by the main executable, as well as by
# the tests.
ADD_LIBRARY(libmepbm
            OBJECT
            histogram.cpp
            models.cpp
			statistics.cpp)

#########################################
### Describe the main executable and from which files
### it needs to be compiled.
ADD_EXECUTABLE(mepbm
	       main.cpp)
TARGET_LINK_LIBRARIES(mepbm
                      $<TARGET_OBJECTS:libmepbm>)


#########################################
### Set up the testsuite machinery

ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)
