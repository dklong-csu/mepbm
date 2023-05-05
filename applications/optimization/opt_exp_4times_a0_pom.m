clearvars
clc
close all

diary opt_exp_4times_a0_pom.log

rng default

p = gcp('nocreate');
if isempty(p)
    parpool(96);
elseif p.NumWorkers < 96
    delete(gcp);
    parpool(96);
end

%%
post_filename = "thinned_posterior_samples.mat";
totalSampsPerDataset = 500;%1000

A0      = optimizableVariable('A0',[1e-4, 1e-2],'Transform','log','Optimize',true);
POM0    = optimizableVariable('POM0',[1e-4, 1e-1],'Transform','log','Optimize',true);
Solvent = optimizableVariable('Solvent',[1, 11.3],'Transform','none','Optimize',false);
Time0   = optimizableVariable('Time0',[0.0167, 24],'Transform','none','Optimize',true);
Time1   = optimizableVariable('Time1',[0.0167, 24],'Transform','none','Optimize',true);
Time2   = optimizableVariable('Time2',[0.0167, 24],'Transform','none','Optimize',true);
Time3   = optimizableVariable('Time3',[0.0167, 24],'Transform','none','Optimize',true);

designVars = [A0, POM0, Solvent, Time0, Time1, Time2, Time3];

optFcn = @(expDesign) -calculateMutualInformation(expDesign,post_filename,totalSampsPerDataset);
stopFcn = @(results,state) stopBayesianOptimization(results,state,50,0.001);

max_cost_evals = 500;%50
% minimizer_filename = strcat(scenario,'.out');
acquisitionFcn = 'expected-improvement-plus';
results_filename = "opt_exp_4times_a0_pom.mat";

pltFcns = {@plotAcquisitionFunction, @plot1DBayesOptIterations}; % 1D only

results = bayesopt(optFcn, designVars, ...
  'PlotFcn', pltFcns,...
  'MaxObjectiveEvaluations',max_cost_evals,...
  'Verbose',1,...
  'XConstraintFcn',@areTimesInOrder,...
  'AcquisitionFunctionName',acquisitionFcn,...   
  'SaveFileName',results_filename,...   
  'OutputFcn',{@saveToFile, stopFcn});
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
            

function isFeasible = areTimesInOrder(expDesign)
% hardcoded for now beacause I only plan on doing max 4 times for thesis
isFeasible = true(size(expDesign));
try
    t0 = expDesign.Time0;
    try
        t1 = expDesign.Time1;
        isFeasible = t1 > t0;
        try
            t2 = expDesign.Time2;
            isFeasible = isFeasible && t2 > t1;
            try 
                t3 = expDesign.Time3;
                isFeasible = isFeasible && t3 > t2;
            catch
            end
        catch
        end
    catch
    end
catch
end
end