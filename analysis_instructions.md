# Instructions for creating a new analysis

These instructions assume the project has already been correctly installed. To check this is the case, from the installed directory `/path/to/install/mepbm/` run the following
```
cmake . -DCMAKE_BUILD_TYPE=release
make -j2
ctest -j2
```
Or if you have more than 2 threads on your CPU, you can make this process faster by specifying, for example, `-j6`. If all executables compile from `make` and all tests pass from `ctest` then the following instructions are ready to be followed.

For the remainder of this document, I will refer to file locations with `.../mepbm/` instead of `/path/to/install/mepbm/` for brevity. Just fill in the file structure in place of the `...` as appropriate.

## Quickstart

Once everything is compiled and setup according to the `Create new analyses directory` section, do the following for an analysis
- Compile the C++ code by running
```
cd .../mepbm/
cmake . -DCMAKE_BUILD_TYPE=release
cd ./applications/name-of-new-directory/
make -j2
```
- Run the Bayesian inversion algorithm with `nohup nice -19 matlab -batch "mcmc; exit" &`
- Once that completes, run `postprocess.m` through the Matlab GUI
  - After the postprocess, you want to see a Gelman-Rubin MPSRF statistic of 1.01 or smaller to be absolutely sure that you can use your results. 
  - If you get a statistic of, say, 1.10 then increase the number of samples (by a factor of 10 if you can wait that long, otherwise use your discretion), change the seed, and use that previous run for the starting points in the next analyses.
- Solve the ODE for the MAP parameters by running `./solve_ode_<extension name> plot_MAP_prm.txt` (similar for mean parameters)
- Plot the MAP/mean parameters against the data by running `plot_sim_against_data.m` in the Matlab GUI. 

## Create new analyses directory

There is an example directory setup to make things faster. This can be found in `.../mepbm/applications/example`. This directory contains templates for everything you need to run a full Bayesian inversion analysis. Create a new directory for your own analysis by running the following commands from the `.../mepbm/applications/example/` directory:

```
cd /path/to/install/dir/mepbm/applications/example/
cp calc_log_likelihood.cpp CMakeLists.txt hpo4.inp hpo4.inp.tpl hpo4.out mcmc.m parse_numeric_output.m plot_sim_against_data.m postprocess.m solve_ode.cpp ../name-of-new-directory
cd ../name-of-new-directory
```

Now in `.../mepbm/applications/` there is a file called `CMakeLists.txt` with a few commands of the form `ADD_SUBDIRECTORY(...)`. To this file add the line `ADD_SUBDIRECTORY(name-of-new-directory)`. This will allow you to compile the new programs you create the next time you run `cmake` and `make`.

### CMakeLists.txt

In `.../mepbm/applications/name-of-new-directory` there is a file also called `CMakeLists.txt`. It should have the following
```
CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE(Threads)

# Change the executable name to something meaningful
ADD_EXECUTABLE(calc_log_likelihood_example
    calc_log_likelihood.cpp)
TARGET_LINK_LIBRARIES(calc_log_likelihood_example
    libmepbm)

# Change the executable name to something meaningful
ADD_EXECUTABLE(solve_ode_example
    solve_ode.cpp)
TARGET_LINK_LIBRARIES(solve_ode_example
    libmepbm)
```
CMake does not allow the creation of executables with the same name even if they are in different directories. So change all instances of `calc_log_likelihood_example` and `solve_ode_example` to different, unique names. For example, for a Cadmium-Selenide system, one might use the names `calc_log_likelihood_CdSe` and `solve_ode_CdSe`.

### mcmc.m

The file `mcmc.m` is what is used to initiate a Bayesian inversion analysis. This requires a compiled executable that calculates the log likelihood given a set of parameters. Matlab communicates with this executable using an input and output file. Based on your application, you will want to make the following changes
- Lines 9-11 -- Adjust the variable names as lower and upper bounds (`lb`, `ub`) as necessary.
- Line 14 -- Change the seed for the random number generator if desired (or if doing subsequent runs, the seed should change).
- Line 16 -- Choose the number of desired samples (more is better)
- Line 17 -- If a previous run was performed and you want to draw initial samples from that distribution, provide the path to that file (should be a `.mat` file saved by a previous run of `mcmc.m`).
- Line 20 -- Enter the name of the executable that calculates the log likelihood.
- Line 21 -- Enter the base name of the files that interface between Matlab and the executable. E.g. if you enter `uq_link_root = 'myFile';` then you need to have the files `myFile.inp`, `myFile.inp.tpl`, and `myFile.out` in this directory with contents compatible with the description in the `Input Files` section.

This code can be run in the Matlab GUI or can be set to run in the background (e.g. on a server) with a command similar to
```
nohup nice -19 matlab -batch "mcmc; exit" &
```
which will run the command in the background (i.e. you can close the terminal you launched the code from), `nice -19` will give the program the lowest CPU priority for the scheduler so that if you are running something for a long time on a shared computing resource, you do not hog the CPU. This can simply be left out of the command if this does not matter to you. `-batch` launches Matlab without the GUI and without splash. `nohup` is the program that launches your program in the background and any command line output that would normally appear will be printed in the file `nohup.out`. If you want the output to be saved in a different file, you may instead run the command
```
nohup nice -19 matlab -batch "mcmc; exit" > my_file_name.out &
```

### Input Files

The example files used to interface between Matlab and the executable are called `hpo4.inp`, `hpo4.inp.tpl`, and `hpo4.out`. The root names of these can be whatever you like as long as you adjust the name in `mcmc.m` as described in the section for that file. So, for example, you could perform
```
mv hpo4.inp myFile.inp
mv hpo4.inp.tpl myFile.inp.tpl
mv hpo4.out myFile.out
```
`hpo4.inp` should have a single number per row for each parameter in your analysis. The actual number does not matter since `mcmc.m` will take care of populating the file appropriately. `hpo4.inp.tpl` is a template file that tells the UQLab software how to write a file compatible with the executables. This one is important to be specific with the contents. The easiest way to write this is to have a single line for each parameter in the analysis. The first line should read `<X0001>`, the second line `<X0002>`, and so on. For example, if you have 15 parameters, your file will look like
```
<X0001>
<X0002>
...
<X0014>
<X0015>
```
Finally, `hpo4.out` simply needs to exist, so it can be empty.

### postprocess.m

This script creates a number of plots to summarize the results of the run and creates two files `plot_MAP_prm.txt` and `plot_mean_prm.txt`, which writes the parameters corresponding to the maximum a posteriori parameters (the best fit) and the mean parameters. These can be used with `solve_ode.cpp` and `plot_sim_against_data.m` to plot the graph corresponding to the MAP or mean parameters against the data.

The only modifications you should need to make are
- Line 6 -- Change the file name to the analysis you want to postprocess.
- Line 24 -- Change how much burn-in you want (i.e. throw away the first N samples before summarizing). A value in (0,1) not including 0,1 means that proportion of samples are discarded (e.g. 0.5 means 50% of samples are discarded). An integer value N of 1 or more means that the first N samples are discarded. 

If you find certain plots are not useful, you can deactivate them by:
- Prior and posterior plots (you cannot turn off the prior plots without turning off the posterior plots as far as I am aware) -- you want these, so leave line 27 `'scatterplot', 'all'` alone.
- Sample number vs Mean parameter value -- delete line 28 `'meanConvergence','all', ...`
- Acceptance ratio of each chain -- delete line 29 `'acceptance',true, ...`
- Plot of parameter value vs sample number along with the associated marginal distribution for each parameter -- delete line 30 `'trace','all'`

### plot_sim_against_data.m

This script plots the simulated particle size distribution against the collected data. Modifications needed here are
- Line 2 -- Change to the Matlab file that contains all of the data
- Line 5 -- Change to the file that `solve_ode.cpp` created
- Line 9 -- Input the particle sizes that your simulation corresponds to
- Line 11 -- Change the formula that converts from particle size to diameter if necessary
- Lines 16-23 -- This is where plots are created. For the piece of code `plot_tem(S2,tem_sol0,particle_diameters, s);` consider the following
  - `S2` is the name of the data set for the first time point. Adjust this based on the naming convention you have.
  - `tem_sol0` is the name of the variable generated from `solve_ode.cpp`. In general, the simulation for time point i will be in the variable `tem_sol{i-1}` where `{i-1}` just means the evaluation of that expression.
  - Add as many of these as you have time points and make sure to adjust the line underneath `title(...)` as you deem fit to title the plot.
- Line 31 -- Change this to have the correct bins. The example has bins from 0.4nm to 2.3nm with width 0.5nm (hence (2.3-0.4)/0.5=38 bins).
- Lines 38-39 -- Adjust the limits of the plots as necessary.
- If particles grow by 2, then lines 33 and 34 are helpful to make the plot look nicer. If your particles grow by more than 2 and the plots look odd, then something more involved might be necessary and I am happy to discuss. If your particles grow by 1, then simply change to
```
sol_avg = conc_transform;
diam_avg = diameters;
```

### calc_log_likelihood.cpp

This is the main code that calculates the log likelihood. There are comments throughout the code indicating where you might need to change code for your situation. Some things to note
- Line 9-10 -- This is what determines what kind a matrices are used and how linear systems are solved. What's in there by default is a good choice for systems of a few hundred particles and larger with limited agglomeration. If your system is small (e.g. 50-100 particles) you might be better off with using dense matrices instead of sparse matrices and then a corresponding dense linear solver. For example, you might try
```
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Solver = Eigen::PartialPivLU< Matrix >;
```
and see if you can solve the ODE faster. This can be tested with
```
time ./calc_log_likelihood_... my_parameters.txt
... change code to use a different solver ...
time ./calc_log_likelihood_... my_parameters.txt
```
This can be done for a number of different linear solvers if you really need to. Ones I have tested can be found in `.../mepbm/tests/sunlinearsolver01.cc`-`.../mepbm/tests/sunlinearsolver09.cc`. Depending on the method, line 66 of `calc_log_likelihood.cpp` will need to change. The possibilities are `create_sparse_iterative_solver`, `create_sparse_direct_solver`, and `create_dense_direct_solver`. If you are unsure what type of solver you want to use, you can look at `.../mepbm/tests/sunlinearsolver0X.cc` to find the test corresponding to the solver you want to use and use the function that test uses.
- Lines 2-3 -- Replace these headers with the ones corresponding to your nanoparticle system.

### solve_ode.cpp

This is almost exactly the same as `calc_log_likelihood.cpp` except it exports the ODE solution during the process of the log likelihood calculation. This occurs in lines 140-146. If you modify `calc_log_likelihood.cpp` you can essentially copy-past that into `solve_ode.cpp` and then add the export lines into the code.