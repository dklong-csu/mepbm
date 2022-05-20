# Code required
these are the code changes needed for a new analysis

# Run instructions
1. On a local machine, just run `mcmc.m` in Matlab normally.
  - On a server, run the command `nohup nice -19 matlab -batch "mcmc; exit" &`
  - The results are
    1. A file `nohup.out` containing information about the run.
    2. A file of the form `BI_seed_{num1}_ver_{num2}.mat` containing the Bayesian inversion results.
2. On a local machine, run the script `postprocess.m` through Matlab.
  - Numerous graphs are created. These can be saved through the GUI if desired.
  - Statistics are generated in the Matlab Command Window. One thing to look for indicating covergence is Gelman-Rubin MPSRF. Theoretical convergence with infinite samples results in a GR score of 1, so a metric less than 1.1 or maybe even 1.01 is what should be targeted.
  - This script also creates two files `plot_MAP_prm.txt` and `plot_mean_prm.txt` which provide the parameters of the mean of the run and the parameters corresponding to the maximum posterior value.
3. Run `./solve_ode plot_mean_prm.txt` and `./solve_ode plot_MAP_prm.txt`.
  - This solves the ODEs for those parameters of interest and outputs the results so they can be graphed alongside the data.
4. Run `plot_sim_against_data.m` for each of the files created in the previous step to see the simulation plotted against the data.u