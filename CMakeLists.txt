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
### Describe the main executable and from which files
### it needs to be compiled.
ADD_EXECUTABLE(mepbm
	       main.cpp)



#########################################
### Set up the testsuite machinery

ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)
