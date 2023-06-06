%% User input
% Make changes to this section
close all
clearvars
clc

addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
uqlab -nosplash

% Parameter information
var_names = ["kf3","kb3","k13","k23","k33","M3","kf4","kb4","k14","k24","k34","k44","M4","w"];
% [.5 .5] gives equal probability to each model
% [.8 .1] gives 9% probability to 4-step and 91% to 3-step
beta_prm = [.5 .5];

myBI3 = extractAnalysis("results/normal_approx/Results_3step_MHv2_ver_4.mat");
myBI4 = extractAnalysis("results/normal_approx/Results_4step_AMv2_ver_3.mat");


% Sampler settings
seed = 1;
n_chains = 20; 
n_samples = 5000;

tuning_samples = 100; %100
tuning_acceptance_range = [.2 .5];
max_tuning_iters = 0;

% File names
ll_exec_name = 'calc_likelihood_model_select';
uq_link_root = "pomModelSelectSym";
analysis_name = "Results_Model_Select_Sym";


%% Make prior distribution from 3-step and 4-step data
map3 = myBI3.Results.PostProc.PointEstimate.X{1};
sigma3 = myBI3.Results.PostProc.Dependence.Cov;
corr3 = myBI3.Results.PostProc.Dependence.Corr;

map4 = myBI4.Results.PostProc.PointEstimate.X{1};
sigma4 = myBI4.Results.PostProc.Dependence.Cov;
corr4 = myBI4.Results.PostProc.Dependence.Corr;

[r3, c3] = size(corr3);
[r4, c4] = size(corr4);

sigma = zeros(r3+r4,c3+c4);
sigma(1:r3,1:c3) = sigma3;
sigma(r3+1:r3+r4,c3+1:c3+c4) = sigma4;

map = [map3 map4];

params = ["kf3","kb3","k13","k23","k33","M3","kf4","kb4","k14","k24","k34","k44","M4","omega"];
for i=1:length(params)-1
%     Input.Marginals(i).Name = params(i);
    Input.Marginals(i).Type = 'Gaussian';
    Input.Marginals(i).Moments = [map(i) sqrt(sigma(i,i))];
    if i==6 || i==13
      Input.Marginals(i).Bounds = [3 600];
    else
      Input.Marginals(i).Bounds = [0 Inf];
    end
end
Input.Marginals(length(params)).Name = var_names(end);
Input.Marginals(length(params)).Type = 'Beta';
Input.Marginals(length(params)).Parameters = beta_prm;
Input.Marginals(length(params)).Bounds = [0 1];

corr = zeros(length(params));

corr(1:r3,1:c3) = corr3;
corr(r3+1:r3+r4,c3+1:c3+c4) = corr4;
corr(end,end) = 1;

% quick fix 
%corr(7,8) = .5;
%corr(8,7) = .5;

Input.Copula.Type = 'Gaussian';
Input.Copula.Parameters = corr;

myPriorDist = uq_createInput(Input);

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
Solver.MCMC.Steps = tuning_samples;
BayesOpts.Solver = Solver;

sigma_prop = corr * 0;
sigma_prop(1:end-1, 1:end-1) = sigma;
sigma_prop(end,end) = 1;

%% Log file to save results
log_name = strcat("Log_",uq_link_root,".txt");
logID = fopen(log_name,'w');


% Bisection method to seek optimal proposal scale
ar_check_passed = false;
scale_min = -6;
scale_max = 6;
scale = -3;
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
    Solver.MCMC.Proposal.Cov = (10.^scale)*sigma_prop;
    % Solver.MCMC.Proposal.Cov(end,end) = sigma_prop(end,end);
    BayesOpts.Prior = myPriorDist;

    BI = uq_createAnalysis(BayesOpts,'-private');
    delete(strcat(uq_link_root,"*"));
    [ar_check_passed, q] = acceptance_ratio_check(BI, tuning_acceptance_range, 0.7);
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


rng(seed,'twister')
Solver.MCMC.Steps = n_samples;
Solver.MCMC.Proposal.Cov = (10.^scale)*sigma_prop;
BayesOpts.Solver = Solver;

BayesOpts.Prior = myPriorDist;

% Run analysis
BI = uq_createAnalysis(BayesOpts,'-private');

% Clean up created files that I don't want
delete(strcat(uq_link_root,"*"));



%% Save results
save_BI_results(BI, analysis_name, 1)

uq_postProcessInversion(BI, ...
     'pointEstimate', {'MAP','Mean'}, ...
     'burnIn', 1, ...
     'gelmanRubin',true)

uq_print(BI)

fclose(logID);

%%
fprintf("%%-----------------Acceptance ratio quantiles\n")
q=quantile(BI.Results.Acceptance,[.025 .25 .5 .75 .975]);
fprintf(" 2.5%%    25%%    50%%    75%%  97.5%%\n")
fprintf("-----------------------------------\n")
fprintf("%5.3f  ",q)
fprintf("\n")

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




%%
function result = extractAnalysis(file)
    load(file, "BI");
    result = BI;
end