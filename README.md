Mechanism-Enabled Population Balance Modeling (ME-PBM)
======================================================

ME-PBM is an approach to modeling nanoparticle formation through continuous
nucleation (precursor becoming a particle), growth (precursor combining with a particle),
and agglomeration (two particles combining) processes. This library allows for 
the mathematical description of these processes in terms of ordinary differential equations (ODEs).
Moreover, this library is capable of accepting custom processes to form a unique mechanism if so desired.

This repository also houses the application code used in papers our group is working on. This primarily includes
using [Bayesian Inversion](https://en.wikipedia.org/wiki/Inverse_problem#Bayesian_approach) to estimate the 
parameters describing nanoparticle formation mechanisms as well as their confidence intervals.

This code is written in C++14 with visualization for papers done in Matlab.

## Installation

### Dependencies

This library has a few dependencies that are necessary to install. Furthermore, this is a [CMake](https://cmake.org/)
project, so make sure that is installed on your machine.

The main library depends on the [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page) for
handling linear algebra. Installation can be performed with

```
  git clone https://gitlab.com/libeigen/eigen.git
  mkdir ./eigen/build
  mkdir ./eigen/install
  cd ./eigen/build
  cmake ../ -DCMAKE_INSTALL_PREFIX=../install
  make install
```

Performing the Bayesian Inversion utilizes the [SampleFlow](https://github.com/bangerth/SampleFlow) library. 
This is a header-only library, so installation is easy

```
  git clone https://github.com/bangerth/SampleFlow.git
```

You can use [Doxygen](https://www.doxygen.nl/index.html) to produce documentation and [Numdiff](https://www.nongnu.org/numdiff/)
is used in the test suite

```
  sudo apt-get install doxygen numdiff
```

or similar for non-Ubuntu users.

Differential equations are solved using [SUNDIALS](https://computing.llnl.gov/projects/sundials). This can be installed by navigating to where you want to install this software and performing
```
mkdir sundials
cd ./sundials
wget https://github.com/LLNL/sundials/releases/download/v5.7.0/sundials-5.7.0.tar.gz
tar zxf sundials-5.7.0.tar.gz
cd sundials-5.7.0
mkdir ./build
mkdir ./install
cd ./build
cmake ../ -DCMAKE_INSTALL_PREFIX=../install
make install
```

In total, all of the C++ dependencies can be installed (perhaps with minor modification to paths based on your preferences) by performing
```
sudo apt-get install --fix-missing doxygen graphviz libboost-all-dev numdiff -y
doxygen --version
git clone https://github.com/bangerth/SampleFlow.git
git clone https://gitlab.com/libeigen/eigen.git
mkdir ./eigen/build
mkdir ./eigen/install
cd ./eigen/build
cmake ../ -DCMAKE_INSTALL_PREFIX=../install
make install
cd ../../
mkdir sundials
cd ./sundials
wget https://github.com/LLNL/sundials/releases/download/v5.7.0/sundials-5.7.0.tar.gz
tar zxf sundials-5.7.0.tar.gz
cd sundials-5.7.0
mkdir ./build
mkdir ./install
cd ./build
cmake ../ -DCMAKE_INSTALL_PREFIX=../install
make install
```

Where the first line `sudo apt-get install --fix-missing doxygen graphviz libboost-all-dev numdiff -y` will need to be modified based on what operating system you use (these instructions are for Ubuntu).

Lastly, a Matlab package called [UQLab](https://www.uqlab.com/) is used for most of the statistics. You need to register (free) on their website, download the software, and install it. Then the provided scripts and examples will be able to use the UQLab functions. The installation is simple: download UQLab and extract the archive whereever you like. Then navigate to `/path/to/uqlab/install/core`. Open Matlab in that directory and run `uqlab_install.m`. To then test to make sure everything installed correctly, run `uqlab -selftest` from that same Matlab session. This will take 30 minutes or so.

### Main library

Once in the ME-PBM directory, perform
```
  cmake . -DSAMPLEFLOW_DIR=/path/to/SampleFlow -DEIGEN_DIR=/path/to/eigen/install -DSUNDIALS_DIR=/path/to/sundials/sundials-5.7.0/install  
  make
```

For analysis runs, optimization can be turned on by with `cmake . -DCMAKE_BUILD_TYPE=release && make` whereas debug mode can be turned with `cmake . -DCMAKE_BUILD_TYPE=debug && make`

## Testing

There are many tests in the `tests/` directory. Note that optimizations need to be turned on to pass all of the tests (and for the to not take a very long time). To execture them, do

```
cmake . -DCMAKE_BUILD_TYPE=release
ctest
```
