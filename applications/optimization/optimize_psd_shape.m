%% 
clc
clearvars
close all

% name the log file
scenario = 'optimize_psd_shape_a0_tend';
diary(strcat(scenario,'.log'))

rng('default')

%% Decide which variables to optimize

A0 = optimizableVariable('A0',[1e-4, 1],'Transform','log','Optimize',true);
tend = optimizableVariable('tend',[0.1, 100],'Transform','log','Optimize',true);
drip_rate = optimizableVariable('drip_rate',[1e-10, 1],'Transform','log','Optimize',false);
drip_time = optimizableVariable('drip_time',[0.1, 30],'Transform','none','Optimize',false);
POM0 = optimizableVariable('POM0',[1e-4, 1],'Transform','log','Optimize',false);
solvent = optimizableVariable('solvent',[1, 11.3],'Transform','none','Optimize',false);
kf_mult = optimizableVariable('kf_mult',[0.1, 10],'Transform','log','Optimize',false);
k1_mult = optimizableVariable('k1_mult',[0.1, 10],'Transform','log','Optimize',false);
k2_mult = optimizableVariable('k2_mult',[0.1, 10],'Transform','log','Optimize',false);
k3_mult = optimizableVariable('k3_mult',[0.1, 10],'Transform','log','Optimize',false);

opt_vars = [A0, tend, drip_rate, drip_time, POM0, solvent, kf_mult, k1_mult, k2_mult, k3_mult];

%%
cost_is_deterministic = false;
posterior_samp_file = "thinned_posterior_samples.mat";
n_samps_per_eval = 500;
max_cost_evals = 500;
desired_particle_size = atoms_to_diam(150);
minimizer_filename = strcat(scenario,'.out');
acquisitionFcn = 'expected-improvement-plus';
results_filename = strcat(scenario,'.mat');

pltFcns = {@plotAcquisitionFunction, @plot1DBayesOptIterations};
stopFcn = @(results,state) stopBayesianOptimization(results,state,50,0.001);
%% Perform BO
if cost_is_deterministic
    data = load(posterior_samp_file);
    stacked_samps = data.stacked_samps;
    draws = draw_from_kde(stacked_samps, n_samps_per_eval);
    fileID = fopen('samples4cost.txt','w');
    for ii=1:size(draws,1)
        d = draws(ii,:);
        for jj=1:length(d)
            fprintf(fileID, "%.20f",d(jj));
            if jj < length(d)
                fprintf(fileID,", ");
            else
                fprintf(fileID,"\n");
            end
        end
    end
    fclose(fileID);

    opt_fcn = @(X) cost_func_shape(X, desired_particle_size, "./calc_cost_function", "BO_cost.inp", "BO_cost.out");
else
    opt_fcn = @(X) cost_func_shape_stochastic(X, desired_particle_size,...
        "./calc_cost_function", "BO_cost.inp", "BO_cost.out",...
        posterior_samp_file, n_samps_per_eval);
end


results = bayesopt(opt_fcn, opt_vars, ...
  'PlotFcn', pltFcns,...
  'IsObjectiveDeterministic', cost_is_deterministic,...
  'MaxObjectiveEvaluations',max_cost_evals,...
  'Verbose',1,...
  'AcquisitionFunctionName',acquisitionFcn,...   
  'SaveFileName',results_filename,...   
  'OutputFcn',{@saveToFile, stopFcn});

%% write best parameters to file
minimizer = results.XAtMinEstimatedObjective;
writeOptPrm2File(minimizer, desired_particle_size, minimizer_filename);


%%
diary off

%%
function stop = stopBayesianOptimization(results,state,min_iters, tol)
stop = false;
switch state
    case 'iteration'
        if results.NumObjectiveEvaluations > min_iters
            [mu, sigma] = predictObjective(results, results.NextPoint);
            best_obj = results.EstimatedObjectiveMinimumTrace(end);
            expect_improve = (best_obj - mu)* normcdf((best_obj - mu)/sigma) ...
                    + sigma * normpdf((best_obj - mu)/sigma);
            perc_improve = abs(expect_improve) / abs(best_obj);
            stop = perc_improve < tol;
            if stop
                fprintf("Optimization terminated because the expected improvement is below tolerance: %f < %f\n",perc_improve,tol)
            end
        end
end
end