%% User input
% Make changes to this section
clearvars
clc

addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
uqlab -nosplash

% Parameter information
var_names = ["kf", "kb", "k1", "k2", "k3", "M"];

% Sampler settings
seed = 1;
n_chains = 20; 
p = gcp('nocreate');
if isempty(p)
    parpool(n_chains);
elseif p.NumWorkers < n_chains
    delete(gcp);
    parpool(n_chains);
end


n_samples = 15000; %15000
previous_run_file = 'results/normal_approx/Results_3step_AM_ver_9.mat';

% File names
% ll_exec_name = 'calc_likelihood_3step';
% uq_link_root = "pom3step";
analysis_name = "Results_eMoM_3step_MH";

%% Perform MCMC
% Most likely don't make changes below!
rng(seed,'twister')

% Likelihood function
myLogLikelihood = @(params,y) custom_log_likelihood_eMoM(params);

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Proposal distribution
% I use previous results here
% if you don't have previous results yet, you could
% (1) do a normal optimization to find "best" point
%     then numerically evaluate Hessian matrix, H, of the log likelihood
%     then set proposal distribution to have covariance matrix = H^-1
%     simulate 100 samples and check acceptance ratio (AR)
%     if AR > ~0.45, set proposal to a*H^-1 where a > 1
%     if AR < ~0.15, set proposal to a*H^-1 where a < 1
%     iterate this process until all chains have AR in ~[0.15, 0.45]
% (2) manually set the proposal to have variance (diagonal of covar matrix)
%     proportional to typical parameter values and do the AR test above
%     e.g. typical parameter value is 1e5, then maybe let variance be 1e8
%     which means standard deviation is 1e4
%     option (1) is probably easier to "tune" with and I would recommend
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
%
% If you don't have previous results, you can do either
% (1) if you did normal optimization and found x* as the optimal
%     parameters then you could make the prior distribution be
%     prior ~ N(x*, H^-1)
%     where N() is normal distribution
%     and H^-1 is the inverse Hessian matrix (numerically estimated)
%     of the likelihood evaluated at x*
map = prevBI.Results.PostProc.PointEstimate.X{1};
corr = prevBI.Results.PostProc.Dependence.Corr;

for i=1:length(var_names)
    Input.Marginals(i).Name = var_names(i);
    Input.Marginals(i).Type = 'Gaussian';
    Input.Marginals(i).Moments = [map(i) sqrt(sigma(i,i))];
    if i==6
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
delete("pom3step*");


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