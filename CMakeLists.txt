CMAKE_MINIMUM_REQUIRED (VERSION 3.1)
PROJECT (mepbm CXX)

#########################################
### Set up compiler flags and input paths

# Compile everything with C++11
SET (CMAKE_CXX_STANDARD 11)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)
SET (CMAKE_CXX_EXTENSIONS OFF)

# Also make sure we link with the threads library in question
FIND_PACKAGE (Threads)

# Find include files in the current directory
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})


#########################################
### Find the SampleFlow library
FIND_PATH(_sampleflow_include_dir
          NAMES sampleflow/consumer.h
          HINTS ${SAMPLEFLOW_DIR}/include)
IF ("${_sampleflow_include_dir}" STREQUAL "_sampleflow_include_dir-NOTFOUND")
  MESSAGE(FATAL_ERROR
          "The SampleFlow library was not found. You have to specify a path "
          "to that library by setting the SAMPLEFLOW_DIR environment variable, "
          "or passing '-DSAMPLEFLOW_DIR=...' as an argument to 'cmake'.")
ELSE()
  MESSAGE(STATUS "Found SampleFlow headers at ${_sampleflow_include_dir}")
  INCLUDE_DIRECTORIES(${_sampleflow_include_dir})
ENDIF()


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
	       main.cpp
               $<TARGET_OBJECTS:libmepbm>)
TARGET_LINK_LIBRARIES (mepbm
                       Threads::Threads)


#########################################
### Set up the testsuite machinery

ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)
