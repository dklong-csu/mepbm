%% User input
% Make changes to this section
clearvars
clc

addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
uqlab -nosplash

% Parameter information
var_names = ["kf", "kb", "k1", "k2", "k3", "k4", "M"];
lb = [0, 0, 0, 0, 0, 0, 3]; % lower bound for the prior 
ub = [1e3, 2e8, 1e8, 1e8, 1e8, 1e8, 600]; % upper bound for the prior

% Covariance matrix
sigma = [4.23077e-05  0 0 0 0 0 0 ; 
0  1.69231e+08  0 0 0 0 0  ;
0 0  1.96752e+08  0 0 0 0  ;
0 0 0  1.35795e+06  0 0 0;  
0 0 0 0  2.87694e+05  0 0  ;
0 0 0 0 0  6.22846e+05  0;  
0 0 0 0 0 0  4.60095e+02];
mean = [6.84619e-02  1.36924e+05  7.93860e+04  1.38803e+04  7.23083e+03  1.69801e+03  1.14879e+02];

% Sampler settings
seed = 1;
n_chains = 20; % recommended: 2x # of parameters
n_samples = 15000;
previous_run_file = '';

% File names
ll_exec_name = 'calc_likelihood_4step';
uq_link_root = "pom4step";
analysis_name = "Results_4step_MH";

%% Perform MCMC
% Most likely don't make changes below!
rng(seed,'twister')

% Likelihood function
myLogLikelihood = @(params,y) custom_log_likelihood(ll_exec_name, uq_link_root, params);

% Bayesian Inversion
BayesOpts.Type = 'Inversion';
BayesOpts.LogLikelihood = myLogLikelihood;
BayesOpts.Data.y = [];
BayesOpts.Display = 'verbose';

% Inversion Algorithm
Solver.Type = 'MCMC';
Solver.MCMC.Sampler = 'MH';
Solver.MCMC.NChains = n_chains;
Solver.MCMC.Steps = n_samples;


starting_samps = mvnrnd(mean, 1.1*sigma, Solver.MCMC.NChains)';
Solver.MCMC.Seed = starting_samps;

Solver.MCMC.Proposal.Cov = 0.05*sigma;


BayesOpts.Solver = Solver;

% Prior distribution
PriorOpts.Name = strcat('Prior',num2str(seed));
for ii=1:length(var_names)
    PriorOpts.Marginals(ii).Name = var_names(ii);
    PriorOpts.Marginals(ii).Type = 'Uniform';
    PriorOpts.Marginals(ii).Parameters = [lb(ii) ub(ii)];
end

myPriorDist = uq_createInput(PriorOpts,'-private');
BayesOpts.Prior = myPriorDist;

% Run analysis
BI = uq_createAnalysis(BayesOpts,'-private');

% Clean up created files that I don't want
delete("pom4step*");


%% Save results
save_BI_results(BI, analysis_name, 1)

uq_postProcessInversion(BI, ...
     'pointEstimate', {'MAP','Mean'}, ...
     'burnIn', 1, ...
     'gelmanRubin',true)

uq_print(BI)

%% Function to safely save the file
function save_BI_results(BI, name, n_tries)
    file_name = strcat(name, "_ver_", num2str(n_tries));
    if ~isfile(strcat(file_name,'.mat'))
        fprintf("Saving file %s.mat\n", file_name)
        save(file_name, "BI")
    else
        fprintf("Trying different file name...\n")
        save_BI_results(BI, name, n_tries+1)
    end
end