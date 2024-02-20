Mechanism-Enabled Population Balance Modeling (ME-PBM)
======================================================

# Quick Start
The following instructions work on Ubuntu (tested on Ubuntu 20.04). On Windows, this works on Windows Subsystem for Linux (WSL).

Run the following lines one-by-one in the terminal.

## Install basic packages
```
sudo apt update && sudo apt upgrade && sudo apt full-upgrade
sudo apt-get install --fix-missing build-essential cmake doxygen graphviz libboost-all-dev numdiff -y
git clone https://github.com/dklong-csu/mepbm.git
cd mepbm
```
## Install dependencies
```
git clone https://github.com/bangerth/SampleFlow.git

git clone https://gitlab.com/libeigen/eigen.git
mkdir ./eigen/build && mkdir ./eigen/install && cd ./eigen/build
cmake ../ -DCMAKE_INSTALL_PREFIX=../install
make install

cd ../../ && mkdir sundials && cd ./sundials
wget https://github.com/LLNL/sundials/releases/download/v5.7.0/sundials-5.7.0.tar.gz
tar zxf sundials-5.7.0.tar.gz
cd sundials-5.7.0 && mkdir ./build && mkdir ./install && cd ./build
cmake ../ -DCMAKE_INSTALL_PREFIX=../install
make install
```

## Check the installation
```
cd ../../../

cmake -DSAMPLEFLOW_DIR="$PWD/SampleFlow" -DEIGEN_DIR="$PWD/eigen/install" -DSUNDIALS_DIR="$PWD/sundials/sundials-5.7.0/install" -DCMAKE_BUILD_TYPE=release .

make -j4

ctest --output-on-failure -j4
```
