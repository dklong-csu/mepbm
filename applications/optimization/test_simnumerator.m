clc
clearvars

%%
post_filename = "thinned_posterior_samples.mat";
design = [0.0012, 0, 11.3, 0.5, 0.75, 4.8];
n_draws = 5;

data = simulateData(design, post_filename, n_draws, n_draws+1);

dataDenom = cell2mat(simulateData(design, post_filename, n_draws, 1));