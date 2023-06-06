%% User input
% Make changes to this section
clearvars
clc

addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
uqlab -nosplash

% Parameter information
var_names = ["kf", "kb", "k1", "k2", "k3", "k4", "M"];

% Sampler settings
seed = 1;
n_chains = 20; % recommended: 2x # of parameters
n_samples = 15000;
previous_run_file = 'Results_4step_AM_ver_8.mat';

% File names
ll_exec_name = 'calc_likelihood_4step';
uq_link_root = "pom4step";
analysis_name = "Results_4step_AMv2";

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

prevBI = extractAnalysis(previous_run_file);
uq_postProcessInversion(prevBI, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 0.5)
sigma = prevBI.Results.PostProc.Dependence.Cov;

Solver.MCMC.Proposal.Cov = 0.5*2.38*2.38/length(var_names) * sigma;

BayesOpts.Solver = Solver;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prior distribution
map = prevBI.Results.PostProc.PointEstimate.X{1};
corr = prevBI.Results.PostProc.Dependence.Corr;

for i=1:length(var_names)
    Input.Marginals(i).Name = var_names(i);
    Input.Marginals(i).Type = 'Gaussian';
    Input.Marginals(i).Moments = [map(i) sqrt(sigma(i,i))];
    if i==length(var_names)
      Input.Marginals(i).Bounds = [3 600];
    else
      Input.Marginals(i).Bounds = [0 Inf];
    end
end

Input.Copula.Type = 'Gaussian';
Input.Copula.Parameters = corr;

myPriorDist = uq_createInput(Input);
BayesOpts.Prior = myPriorDist;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

function result = extractAnalysis(file)
    load(file, "BI");
    result = BI;
end