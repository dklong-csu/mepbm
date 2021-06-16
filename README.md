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

This code is written in C++11 with visualization for papers done in Matlab.

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

Performing large-scale statistical simulations like we do in our papers is computationally expensive. It can be helpful
to have [OpenMP](https://www.openmp.org/) installed on your computer as well to assist in this. Examples of how this is used
can be found in `applications/ir-pom-paper/mcmc`.

### Main library

Once in the ME-PBM directory, perform
```
  cmake -DSAMPLEFLOW_DIR=/path/to/SampleFlow -DEIGEN_DIR=/path/to/eigen/install .
  make
```

For generating many samples, optimization can be turned on by adding `-DCMAKE_BUILD_TYPE=release`
and `-DCMAKE_CXX_FLAGS="-march=native -fopenmp"`. Prior to running code such as in `applications/ir-pom-paper/mcmc` we found it
helpful to do the following prior to running the executable

```
  export OMP_PROC_BIND=spread
  export OMP_PLACES=cores
  export OMP_NUM_THREADS=n
```

where `n` is the number of threads on your CPU (or the number you are willing to allocate).

## Testing

There are many tests in the `tests/` directory. To execture them, do

```
  ctest
```