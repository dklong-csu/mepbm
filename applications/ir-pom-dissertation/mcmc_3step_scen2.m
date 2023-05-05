%%
% Scenario 2
% Run 1 - uniform prior
% Run 2 - Beta prior based on average of data and best sim

%%
clc
clearvars
close all

addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
uqlab -nosplash


%% Problem setup
% Parameter information
var_names = ["kf", "kb", "k1", "k2", "k3", "M"];
lb = [0, 0, 0, 0, 0, 3]; % lower bound for the prior 
ub = [1e3, 2e8, 1e8, 1e8, 1e8, 600]; % upper bound for the prior

% Sampler settings
seed = 1;
n_chains = 20; %20
n_samples = 20000; %10000
previous_run_file = '';

tuning_samples = 100; %100
tuning_acceptance_range = [.15 .4];
max_tuning_iters = 10;

% For run #2, a prior is used based on the number of anticipated particles
% in each bin. Specify here the total number of particles used for the
% prior.
prior_obs = 150;

% File names
ll_exec_name = './calc_likelihood_3step';
ll_beta_exec_name = './calc_likelihood_times_beta_3step';
uq_link_root = "pom3step_scen2";
analysis_name = "Results_3step_scen2";
conc_exec_name = './calc_concentrations_3step';

% Starting value (starting points of chains drawn from an area around this)
mean_prm = [3.31e-3, 6.62e3, 1.24e5, 2.6e5, 6.22e3, 107];
sigma = [2.17579e-07  0 0 0 0 0;  
0  8.70315e+05  0 0 0 0  ;
0 0  2.09842e+08  0 0 0  ;
0 0 0  7.34321e+09  0 0  ;
0 0 0 0  4.41241e+05  0  ;
0 0 0 0 0  4.31463e+01];

%% Log file to save results
log_name = strcat("Log_",uq_link_root,".txt");
logID = fopen(log_name,'w');

%% Run 1 -- Metropolis Hastings
fprintf(logID,"================================\n");
fprintf(logID,"Run 1 - Metropolis Hastings\n");
fprintf(logID,"================================\n");
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


% Starting points of chains
starting_samps = mvnrnd(mean_prm, 1.1*sigma, n_chains)';

Solver.MCMC.Seed = starting_samps;

% Prior distribution
PriorOpts.Name = strcat('Prior',num2str(seed));
for ii=1:length(var_names)
    PriorOpts.Marginals(ii).Name = var_names(ii);
    PriorOpts.Marginals(ii).Type = 'Uniform';
    PriorOpts.Marginals(ii).Parameters = [lb(ii) ub(ii)];
end

myPriorDist = uq_createInput(PriorOpts,'-private');


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
    BayesOpts.Prior = myPriorDist;
    
    BI = uq_createAnalysis(BayesOpts,'-private');
    
    delete(strcat(uq_link_root,"*"));
    [ar_check_passed, q] = acceptance_ratio_check(BI, tuning_acceptance_range);
    median_ar = q(2);
    fprintf(logID,"Median acceptance ratio: %f\n",q(2));
    fprintf(logID,"70%% acceptance ratio interval: [%f -- %f]\n",q(1), q(3));
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
delete(strcat(uq_link_root,"*"));

save_BI_results(BI, strcat(analysis_name,"_run1"), 1)

uq_postProcessInversion(BI, ...
     'pointEstimate', {'MAP','Mean'}, ...
     'burnIn', 0.75, ...
     'gelmanRubin',false)

%uq_print(BI)

%% Load in data to generate prior
TEMdata
bin_edges = 1.4:0.1:4.1;
raw_data = {S2, S3, S4, S5};

% particle diameters
s=(3:2500)';
diams = 0.3000805 .* s .^ (1/3);

% Run sim for MAP point in previous run
map = BI.Results.PostProc.PointEstimate.X{1};
prm_file = strcat("MAP_",uq_link_root,".inp");
fileID = fopen(prm_file,'w');
fprintf(fileID,"%.20f\n",map);
fclose(fileID);
conc_file_root = strcat("conc_",uq_link_root, "_time");
system_call = strcat(conc_exec_name, " ",prm_file, " ", conc_file_root);
[status, cmdout] = system(system_call);

% Compute estimated data for each time point
for tt=1:length(raw_data)
    conc_file = strcat(conc_file_root,"-",num2str(tt),".out");
    conc = parse_numeric_output(conc_file);
    prob_sim = sim_concentration_to_bin_probability(conc, diams, bin_edges);
    prob_data = diameters_to_bin_probability(raw_data{tt}, bin_edges);
    
    prob_avg = (prob_data + prob_sim)/2;
    obs = draw_N_from_bins(prob_avg, prior_obs);
    obs_file = strcat("betaObs_",uq_link_root,"-",num2str(tt-1),".txt");
    obsID = fopen(obs_file,'w');
    fprintf(obsID,"%d\n",round(obs));
    fclose(obsID);
end

% diameters for each bin
bin_diams = bin_edges(1:end-1)+0.05;
diams_file = strcat("diams_",uq_link_root,".txt");
diamsID = fopen(diams_file,'w');
fprintf(diamsID,"%f\n",bin_diams);




%% Run 2 -- MH starting from previous results
rng(seed,'twister')

fprintf(logID,"===================================\n");
fprintf(logID,"Run 2 - MH with previous run's info\n");
fprintf(logID,"===================================\n");

% Likelihood function
betaCount_file = strcat("betaObs_",uq_link_root);
myLogLikelihood = @(params,y) custom_log_likelihood_times_beta(...
    ll_beta_exec_name, uq_link_root, betaCount_file, diams_file, params);

% Bayesian Inversion
BayesOpts.Type = 'Inversion';
BayesOpts.LogLikelihood = myLogLikelihood;
BayesOpts.Data.y = [];
BayesOpts.Display = 'verbose';

% Inversion Algorithm
Solver.Type = 'MCMC';
Solver.MCMC.Sampler = 'MH';
Solver.MCMC.NChains = n_chains;

% Best observed parameters are now center of the starting samples
map = BI.Results.PostProc.PointEstimate.X{1};

% Covariance matrix uncorrelated based on order of magnitude of mean
sigma = BI.Results.PostProc.Dependence.Cov;


% Starting points of chains
starting_samps = max(mvnrnd(map, 1.1*sigma,n_chains),0)';
starting_samps(end,:) = max(3, starting_samps(end,:));
Solver.MCMC.Seed = starting_samps;

% Prior distribution
PriorOpts.Name = strcat('Prior',num2str(seed));
for ii=1:length(var_names)
    PriorOpts.Marginals(ii).Name = var_names(ii);
    PriorOpts.Marginals(ii).Type = 'Uniform';
    PriorOpts.Marginals(ii).Parameters = [lb(ii) ub(ii)];
end

myPriorDist = uq_createInput(PriorOpts,'-private');


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
    [ar_check_passed, q] = acceptance_ratio_check(BI, tuning_acceptance_range);
    median_ar = q(2);
    fprintf(logID,"Median acceptance ratio: %f\n",q(2));
    fprintf(logID,"70%% acceptance ratio interval: [%f -- %f]\n",q(1), q(3));
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
delete(strcat(uq_link_root,"*"));

save_BI_results(BI, strcat(analysis_name,"_run2"), 1)

uq_postProcessInversion(BI, ...
     'pointEstimate', {'MAP','Mean'}, ...
     'burnIn', 1, ...
     'gelmanRubin',true)

uq_print(BI)



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
