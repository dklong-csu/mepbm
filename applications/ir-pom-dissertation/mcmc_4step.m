
clc
clearvars
close all

addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
uqlab -nosplash


%% Problem setup
%==============================================================================================
% Parameter information
var_names = ["kf", "kb", "k1", "k2", "k3", "k4", "M"];
lb = [0, 0, 0, 0, 0, 0, 3]; % lower bound for the prior 
ub = [1e3, 2e8, 1e8, 1e8, 1e8, 1e8, 600]; % upper bound for the prior

% Sampler settings
seed = 1;
n_chains = 20; %20
n_samples = 15000; %15000
previous_run_file = '';

tuning_samples = 300; %300
tuning_acceptance_range = [.2 .5];
max_tuning_iters = 10; %10
tuning_CI_percentile = 0.7;

% File names
ll_exec_name = './calc_likelihood_4step';
uq_link_root = "pom4step";
analysis_name = "Results_4step";

% Starting value (starting points of chains drawn from an area around this)
mean_prm = [6.84619e-02  1.36924e+05  7.93860e+04  1.38803e+04  7.23083e+03  1.69801e+03  1.14879e+02];
sigma = [4.23077e-05  0 0 0 0 0 0 ; 
0  1.69231e+08  0 0 0 0 0  ;
0 0  1.96752e+08  0 0 0 0  ;
0 0 0  1.35795e+06  0 0 0;  
0 0 0 0  2.87694e+05  0 0  ;
0 0 0 0 0  6.22846e+05  0;  
0 0 0 0 0 0  4.60095e+02];
%==============================================================================================



%% Log file to save results
%==============================================================================================
log_name = strcat("Log_",uq_link_root,".txt");
logID = fopen(log_name,'w');
%==============================================================================================



%% Run 1 -- Metropolis Hastings
%==============================================================================================
fprintf(logID,"================================\n");
fprintf(logID,"Run 1 - Metropolis Hastings\n");
fprintf(logID,"================================\n");
rng(seed,'twister')

% Basic setup
%----------------------------------------------------------------------------------------------
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
Solver.MCMC.Epsilon = 1e-4;


% Starting points of chains
starting_samps = mvnrnd(mean_prm, 1.1*sigma, n_chains)';

Solver.MCMC.Seed = starting_samps;
%----------------------------------------------------------------------------------------------


% Prior distribution
%----------------------------------------------------------------------------------------------
PriorOpts.Name = strcat('Prior',num2str(seed));
for ii=1:length(var_names)
    PriorOpts.Marginals(ii).Name = var_names(ii);
    PriorOpts.Marginals(ii).Type = 'Uniform';
    PriorOpts.Marginals(ii).Parameters = [lb(ii) ub(ii)];
end

myPriorDist = uq_createInput(PriorOpts,'-private');
BayesOpts.Prior = myPriorDist;
%----------------------------------------------------------------------------------------------


% Proposal Tuning
%----------------------------------------------------------------------------------------------
ar_check_passed = false;
scale_min = -6;
scale_max = 6;
scale = 1;
tuning_iters = 1;
fprintf(logID,"============================\n");
fprintf(logID,"Tuning proposal distribution\n");
fprintf(logID,"============================\n");
while ~ar_check_passed && tuning_iters <= max_tuning_iters
    fprintf(logID,"\n------------\n");
    fprintf(logID,"Iteration %d\n",tuning_iters);
    fprintf(logID,"------------\n");
    rng(seed,'twister')
    scale = mean([scale_min scale_max]);
    fprintf(logID,"Trying covariance scale: %f\n",10.^scale);
    Solver.MCMC.Steps = tuning_samples;
    Solver.MCMC.Proposal.Cov = (10.^scale)*sigma;
    
    BayesOpts.Solver = Solver;
    
    BI = uq_createAnalysis(BayesOpts,'-private');
    
    delete(strcat(uq_link_root,"*"));
    [ar_check_passed, q] = acceptance_ratio_check(BI, tuning_acceptance_range, tuning_CI_percentile);
    median_ar = q(2);
    fprintf(logID,"Median acceptance ratio: %f\n",q(2));
    fprintf(logID,"%.0f%% acceptance ratio interval: [%f -- %f]\n",100*tuning_CI_percentile, q(1), q(3));
    if ~ar_check_passed
        if median_ar < 0.234
            scale_max = scale;
            fprintf(logID,"Decreasing scale.\n");
        else
            scale_min = scale;
            fprintf(logID,"Increasing scale.\n");
        end
    else
        fprintf(logID,"Scale accepted!\n");
    end
    tuning_iters = tuning_iters + 1;
    if ~ar_check_passed && tuning_iters > max_tuning_iters
        fprintf(logID,"Max tuning iterations met. Moving to full analysis.\n");
    end
end
%----------------------------------------------------------------------------------------------


% Full analysis
%----------------------------------------------------------------------------------------------
fprintf(logID,"\n-------------\n");
fprintf(logID,"Full analysis\n");
fprintf(logID,"-------------\n");
rng(seed,'twister')
fprintf(logID,"Using covariance scale: %f\n",10^scale);
Solver.MCMC.Steps = n_samples;
Solver.MCMC.Proposal.Cov = (10^scale)*sigma;

BayesOpts.Solver = Solver;

BI = uq_createAnalysis(BayesOpts,'-private');
%----------------------------------------------------------------------------------------------


% Save and clean up files
%----------------------------------------------------------------------------------------------
delete(strcat(uq_link_root,"*"));

save_BI_results(BI, strcat(analysis_name,"_run1"), 1)

uq_postProcessInversion(BI, ...
     'pointEstimate', {'MAP','Mean'}, ...
     'burnIn', 0.75, ...
     'gelmanRubin',false)
%----------------------------------------------------------------------------------------------


%==============================================================================================



%% Run 2 -- Adaptive Metropolis - adaptation phase
%==============================================================================================
rng(seed,'twister')

fprintf(logID,"===================================\n");
fprintf(logID,"Run 2 - AM adaptation phase\n");
fprintf(logID,"===================================\n");


% Basic setup
%----------------------------------------------------------------------------------------------
% Likelihood function
myLogLikelihood = @(params,y) custom_log_likelihood(ll_exec_name, uq_link_root, params);

% Bayesian Inversion
BayesOpts.Type = 'Inversion';
BayesOpts.LogLikelihood = myLogLikelihood;
BayesOpts.Data.y = [];
BayesOpts.Display = 'verbose';

% Inversion Algorithm
Solver.Type = 'MCMC';
Solver.MCMC.Sampler = 'AM';
Solver.MCMC.NChains = n_chains;
%----------------------------------------------------------------------------------------------


% Use previous BI for starting points of chains
%----------------------------------------------------------------------------------------------
% Best observed parameters are now center of the starting samples
map = BI.Results.PostProc.PointEstimate.X{1};

% Covariance matrix uncorrelated based on order of magnitude of mean
sigma = BI.Results.PostProc.Dependence.Cov;
rho   = BI.Results.PostProc.Dependence.Corr;

% Starting points of chains
starting_samps = max(mvnrnd(map, 1.1*sigma,n_chains),0)';
starting_samps(end,:) = max(3, starting_samps(end,:));
Solver.MCMC.Seed = starting_samps;
%----------------------------------------------------------------------------------------------


% Prior distribution from previous run
%----------------------------------------------------------------------------------------------
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
Input.Copula.Parameters = rho;

myPriorDist = uq_createInput(Input);
BayesOpts.Prior = myPriorDist;
%----------------------------------------------------------------------------------------------


% Proposal tuning
%----------------------------------------------------------------------------------------------
ar_check_passed = false;
scale_min = -6;
scale_max = 6;
scale = 1;
tuning_iters = 1;
fprintf(logID,"============================\n");
fprintf(logID,"Tuning proposal distribution\n");
fprintf(logID,"============================\n");
while ~ar_check_passed && tuning_iters <= max_tuning_iters
    fprintf(logID,"\n------------\n");
    fprintf(logID,"Iteration %d\n",tuning_iters);
    fprintf(logID,"------------\n");
    rng(seed,'twister')
    scale = mean([scale_min scale_max]);
    fprintf(logID,"Trying covariance scale: %f\n",10^scale);
    Solver.MCMC.Steps = tuning_samples;
    Solver.MCMC.Proposal.Cov = (10^scale)*sigma;
    
    BayesOpts.Solver = Solver;
    BayesOpts.Prior = myPriorDist;
    
    BI = uq_createAnalysis(BayesOpts,'-private');
    
    delete(strcat(uq_link_root,"*"));
    [ar_check_passed, q] = acceptance_ratio_check(BI, tuning_acceptance_range, tuning_CI_percentile);
    median_ar = q(2);
    fprintf(logID,"Median acceptance ratio: %f\n",q(2));
    fprintf(logID,"%.0f%% acceptance ratio interval: [%f -- %f]\n",100*tuning_CI_percentile, q(1), q(3));
    if ~ar_check_passed
        if median_ar < 0.234
            scale_max = scale;
            fprintf(logID,"Decreasing scale.\n");
        else
            scale_min = scale;
            fprintf(logID,"Increasing scale.\n");
        end
    else
        fprintf(logID,"Scale accepted!\n");
    end
    tuning_iters = tuning_iters + 1;
    if ~ar_check_passed && tuning_iters > max_tuning_iters
        fprintf(logID,"Max tuning iterations met. Moving to full analysis.\n");
    end
end
%----------------------------------------------------------------------------------------------


% Full analysis
%----------------------------------------------------------------------------------------------
fprintf(logID,"\n-------------\n");
fprintf(logID,"Full analysis\n");
fprintf(logID,"-------------\n");
rng(seed,'twister')
fprintf(logID,"Using covariance scale: %f\n",10^scale);
Solver.MCMC.Steps = n_samples;
Solver.MCMC.Proposal.Cov = (10^scale)*sigma;

BayesOpts.Solver = Solver;
BayesOpts.Prior = myPriorDist;
BI = uq_createAnalysis(BayesOpts,'-private');
uq_print(BI)
%----------------------------------------------------------------------------------------------


% Save data and clean up
%----------------------------------------------------------------------------------------------
delete(strcat(uq_link_root,"*"));

save_BI_results(BI, strcat(analysis_name,"_run2"), 1)

uq_postProcessInversion(BI, ...
     'pointEstimate', {'MAP','Mean'}, ...
     'burnIn', 1, ...
     'gelmanRubin',true)

%----------------------------------------------------------------------------------------------


%==============================================================================================



%% Run 3 -- Adaptive Metropolis - stationary phase
%==============================================================================================
rng(seed,'twister')

fprintf(logID,"===================================\n");
fprintf(logID,"Run 3 - AM stationary phase\n");
fprintf(logID,"===================================\n");


% Basic setup
%----------------------------------------------------------------------------------------------
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
%----------------------------------------------------------------------------------------------


% Use last sample in previous run for starting point
%----------------------------------------------------------------------------------------------

starting_samps = BI.Results.Sample(end,:,:);
dims = size(starting_samps);
starting_samps = reshape(starting_samps,dims(2), dims(3));
Solver.MCMC.Seed = starting_samps;
%----------------------------------------------------------------------------------------------


% Prior distribution is already set in BayesOpts
%----------------------------------------------------------------------------------------------

%----------------------------------------------------------------------------------------------


% Proposal tuning
%----------------------------------------------------------------------------------------------
ar_check_passed = false;
scale_min = -6;
scale_max = 6;
scale = 1;
tuning_iters = 1;
fprintf(logID,"============================\n");
fprintf(logID,"Tuning proposal distribution\n");
fprintf(logID,"============================\n");
while ~ar_check_passed && tuning_iters <= max_tuning_iters
    fprintf(logID,"\n------------\n");
    fprintf(logID,"Iteration %d\n",tuning_iters);
    fprintf(logID,"------------\n");
    rng(seed,'twister')
    scale = mean([scale_min scale_max]);
    fprintf(logID,"Trying covariance scale: %f\n",10^scale);
    Solver.MCMC.Steps = tuning_samples;
    Solver.MCMC.Proposal.Cov = (10^scale)*sigma;
    
    BayesOpts.Solver = Solver;
    BayesOpts.Prior = myPriorDist;
    
    BI = uq_createAnalysis(BayesOpts,'-private');
    
    delete(strcat(uq_link_root,"*"));
    [ar_check_passed, q] = acceptance_ratio_check(BI, tuning_acceptance_range, tuning_CI_percentile);
    median_ar = q(2);
    fprintf(logID,"Median acceptance ratio: %f\n",q(2));
    fprintf(logID,"%.0f%% acceptance ratio interval: [%f -- %f]\n",100*tuning_CI_percentile, q(1), q(3));
    if ~ar_check_passed
        if median_ar < 0.234
            scale_max = scale;
            fprintf(logID,"Decreasing scale.\n");
        else
            scale_min = scale;
            fprintf(logID,"Increasing scale.\n");
        end
    else
        fprintf(logID,"Scale accepted!\n");
    end
    tuning_iters = tuning_iters + 1;
    if ~ar_check_passed && tuning_iters > max_tuning_iters
        fprintf(logID,"Max tuning iterations met. Moving to full analysis.\n");
    end
end
%----------------------------------------------------------------------------------------------


% Full analysis
%----------------------------------------------------------------------------------------------
fprintf(logID,"\n-------------\n");
fprintf(logID,"Full analysis\n");
fprintf(logID,"-------------\n");
rng(seed,'twister')
fprintf(logID,"Using covariance scale: %f\n",10^scale);
Solver.MCMC.Steps = n_samples;
Solver.MCMC.Proposal.Cov = (10^scale)*sigma;

BayesOpts.Solver = Solver;
BayesOpts.Prior = myPriorDist;

BI = uq_createAnalysis(BayesOpts,'-private');
%----------------------------------------------------------------------------------------------


% Save data and clean up
%----------------------------------------------------------------------------------------------
delete(strcat(uq_link_root,"*"));

save_BI_results(BI, strcat(analysis_name,"_run3"), 1)

uq_postProcessInversion(BI, ...
     'pointEstimate', {'MAP','Mean'}, ...
     'burnIn', 1, ...
     'gelmanRubin',true)

uq_print(BI)
%----------------------------------------------------------------------------------------------


%==============================================================================================



% Close log file
fclose(logID);

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
