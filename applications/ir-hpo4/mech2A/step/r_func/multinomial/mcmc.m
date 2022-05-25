%% User input
% Make changes to this section
clearvars
clc

uqlab -nosplash

% Parameter information
var_names = ["kf", "kb", "k1", "k2", "k3", "k4"];
lb = realmin * ones(1,6); % lower bound for the prior 
ub = [1e3, 1e9, 1e6, 1e6, 1e6, 1e7]; % upper bound for the prior

% Sampler settings
seed = 1;
n_chains = 2*length(var_names); % recommended: 2x # of parameters
n_samples = 250000;
previous_run_file = '';

% File names
ll_exec_name = 'calc_log_likelihood_2A_step_r_multi';
uq_link_root = "hpo4";

%% Perform MCMC
% Most likely don't make changes below!
rng(seed,'twister')

% Create the model with UQLink
ModelOpts.Type = 'UQLink';
ModelOpts.Name = strcat('LogLikelihood',num2str(seed));

% Create files based on seed to allow for parallel computation

INPUTFILE = strcat(uq_link_root,"-",num2str(seed),"-.inp");
TEMPLATEFILE = strcat(uq_link_root,"-",num2str(seed),"-.inp.tpl");
OUTPUTFILE = strcat(uq_link_root,"-",num2str(seed),"-.out");

copyfile(strcat(uq_link_root,".inp"),INPUTFILE)
copyfile(strcat(uq_link_root,".inp.tpl"),TEMPLATEFILE)
copyfile(strcat(uq_link_root,".out"),OUTPUTFILE) 

% Describe execution command line
EXECUTABLE = fullfile(pwd, ll_exec_name);
INPUTFILE = strcat(uq_link_root,"-",num2str(seed),"-.inp");
COMMANDLINE = sprintf('%s %s',EXECUTABLE, INPUTFILE);
ModelOpts.Command = COMMANDLINE;

% Template for input file
ModelOpts.Template = strcat(uq_link_root,"-",num2str(seed),"-.inp.tpl");

% Matlab function that parses output file
ModelOpts.Output.Parser = 'parse_numeric_output';
ModelOpts.Output.FileName = strcat(uq_link_root,"-",num2str(seed),"-.out");

% Set execution path
ModelOpts.ExecutionPath = fullfile(pwd);

% Mute display
ModelOpts.Display = 'quiet';

% Ignore file deletion for UQLink
ModelOpts.Archiving.Action = 'ignore';

% Create the UQLink model
myModel = uq_createModel(ModelOpts,'-private');

% Likelihood function
myLogLikelihood = @(params,y) uq_evalModel(myModel, params);

% Bayesian Inversion
BayesOpts.Type = 'Inversion';
BayesOpts.LogLikelihood = myLogLikelihood;
BayesOpts.Data.y = [];
BayesOpts.Display = 'verbose';

% Inversion Algorithm
Solver.Type = 'MCMC';
Solver.MCMC.Sampler = 'AIES';
Solver.MCMC.NChains = n_chains;
Solver.MCMC.Steps = n_samples;

% Load previous run if specified and seed based on that
if ~strcmp('',previous_run_file) && isfile(previous_run_file)
    load(previous_run_file, "BI")
    post_samp = BI.Results.PostProc.PostSample;
    [n_samps_prev, ~, n_chains_prev] = size(post_samp);
    indices = [randi(n_samps_prev,Solver.MCMC.NChains,1) randi(n_chains_prev, Solver.MCMC.NChains,1)];
    starting_samps = zeros(9, Solver.MCMC.NChains);
    for ii=1:Solver.MCMC.NChains
        starting_samps(:,ii) = post_samp(indices(ii,1), :, indices(ii,2));
    end
    Solver.MCMC.Seed = starting_samps;
    
    fprintf("Starting points drawn from posterior of previous run.")
elseif ~strcmp('',previous_run_file)
    fprintf("Specified results of previous run does not exist. Using starting points based on the prior.")
else
    fprintf("No previous run specified. Using starting points based on the prior.")
end



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
delete(strcat(ModelOpts.Name,'.mat'))
delete(INPUTFILE)
delete(TEMPLATEFILE)
delete(OUTPUTFILE)
delete(strcat("hpo4-",num2str(seed),"-*"));


%% Save results
save_BI_results(BI, seed, 1)

uq_postProcessInversion(BI, ...
     'pointEstimate', {'MAP','Mean'}, ...
     'burnIn', 1, ...
     'gelmanRubin',true)

uq_print(BI)

%% Function to safely save the file
function save_BI_results(BI, seed_val, n_tries)
    file_name = strcat("BI_seed_", num2str(seed_val), "_ver_", num2str(n_tries));
    if ~isfile(strcat(file_name,'.mat'))
        fprintf("Saving file %s.mat\n", file_name)
        save(file_name, "BI")
    else
        fprintf("Trying different file name...\n")
        save_BI_results(BI, seed_val, n_tries+1)
    end
end