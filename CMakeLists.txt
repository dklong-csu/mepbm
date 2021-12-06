CMAKE_MINIMUM_REQUIRED (VERSION 3.1)
PROJECT (mepbm CXX)

#########################################
### Set up compiler flags and input paths

# Compile everything with C++11
SET (CMAKE_CXX_STANDARD 14)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)
SET (CMAKE_CXX_EXTENSIONS OFF)

# Also make sure we link with the threads library in question
FIND_PACKAGE (Threads)


#########################################
### Find the Eigen library
FIND_PATH(_eigen_include_dir
          NAMES eigen3/Eigen/Dense
          HINTS ${EIGEN_DIR}/include)
IF ("${_eigen_include_dir}" STREQUAL "_eigen_include_dir-NOTFOUND")
  MESSAGE(FATAL_ERROR
          "The Eigen library was not found. You have to specify a path "
          "to that library by setting the EIGEN_DIR environment variable, "
          "or passing '-DEIGEN_DIR=...' as an argument to 'cmake'.")
ELSE()
  MESSAGE(STATUS "Found EIGEN headers at ${_eigen_include_dir}")
  INCLUDE_DIRECTORIES(${_eigen_include_dir})
ENDIF()



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
### Find the SUNDIALS library
FIND_PACKAGE (
        _sundials
        NAMES SUNDIALS
        HINTS ${SUNDIALS_DIR} 
        REQUIRED
)
IF ("${_sundials}" STREQUAL "_sundials-NOTFOUND")
  MESSAGE(FATAL_ERROR
          "The SUNDIALS library was not found. You have to specify a path "
          "to that library by setting the SUNDIALS_DIR environment variable, "
          "or by passing '-DSUNDIALS_DIR=...' as an argument to 'cmake'.")
ELSE()
  MESSAGE(STATUS "Found SUNDIALS")
ENDIF()


FIND_PATH(_sundials_include_dir
        NAMES cvode/cvode.h
        HINTS ${SUNDIALS_DIR}/include)
IF ("${_sundials_include_dir}" STREQUAL "_sundials_include_dir-NOTFOUND")
  MESSAGE(FATAL_ERROR
          "The SUNDIALS headers were not found. You have to specify a path "
          "to that library by setting the SUNDIALS_DIR environment variable, "
          "or by passing '-DSUNDIALS_DIR=...' as an argument to 'cmake'.")
ELSE()
  MESSAGE(STATUS "Found SUNDIALS headers at ${_sundials_include_dir}")
  INCLUDE_DIRECTORIES(${_sundials_include_dir})
ENDIF()


FIND_LIBRARY(_sundials_cvode_lib
        NAMES sundials_cvode
        HINTS ${SUNDIALS_DIR}/lib)
IF ("${_sundials_cvode_lib}" STREQUAL "_sundials_cvode_lib-NOTFOUND")
  MESSAGE(FATAL_ERROR
          "The SUNDIALS CVODE library was not found. You have to specify a path "
          "to that library by setting the SUNDIALS_DIR environment variable, "
          "or by passing '-DSUNDIALS_DIR=...' as an argument to 'cmake'.")
ELSE()
  MESSAGE(STATUS "Found SUNDIALS CVODE library at ${_sundials_CVode_lib}")
  INCLUDE_DIRECTORIES(${_sundials_include_dir})
  LINK_DIRECTORIES(${SUNDIALS_DIR}/lib)
ENDIF()



#########################################
# Find include files in the current directory
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/lib)



#########################################
# Describe what files should go into the library that contains all of
# the functionality that is used by the main executable, as well as by
# the tests.
#
# There is, at the moment, no such library because everything is in header
# files. So the only thing we can do is define an "interface" library
# that collects what everything depends upon.
ADD_LIBRARY(libmepbm INTERFACE)
TARGET_INCLUDE_DIRECTORIES(libmepbm
                           INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
TARGET_LINK_LIBRARIES(libmepbm
                      INTERFACE
                      Threads::Threads)
TARGET_LINK_LIBRARIES(libmepbm
                      INTERFACE
                      ${_sundials_cvode_lib})



#########################################
### Set up the testsuite machinery

ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)


#########################################
### Set rules to compile applications
ADD_SUBDIRECTORY(applications)
