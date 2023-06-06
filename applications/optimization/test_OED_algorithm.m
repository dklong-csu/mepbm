clearvars
clc
close all

diary opt_exp_design.log

rng default
%%
post_filename = "thinned_posterior_samples.mat";
totalSampsPerDataset = 1000;%1000

A0      = optimizableVariable('A0',[1e-4, 1],'Transform','log','Optimize',false);
POM0    = optimizableVariable('POM0',[1e-4, 1],'Transform','log','Optimize',false);
Solvent = optimizableVariable('Solvent',[1, 11.3],'Transform','none','Optimize',false);
Time0   = optimizableVariable('Time0',[0.1, 10],'Transform','none','Optimize',true);
Time1   = optimizableVariable('Time1',[0.1, 10],'Transform','none','Optimize',true);
Time2   = optimizableVariable('Time2',[0.1, 10],'Transform','none','Optimize',false);
Time3   = optimizableVariable('Time3',[0.1, 10],'Transform','none','Optimize',false);

designVars = [A0, POM0, Solvent, Time0, Time1, Time2, Time3];

optFcn = @(expDesign) -calculateMutualInformation(expDesign,post_filename,totalSampsPerDataset);


max_cost_evals = 50;%50
% minimizer_filename = strcat(scenario,'.out');
acquisitionFcn = 'expected-improvement-plus';
results_filename = "opt_exp_design.mat";

pltFcns = {@plotAcquisitionFunction, @plot1DBayesOptIterations}; % 1D only

results = bayesopt(optFcn, designVars, ...
  'PlotFcn', pltFcns,...
  'MaxObjectiveEvaluations',max_cost_evals,...
  'Verbose',1,...
  'XConstraintFcn',@areTimesInOrder,...
  'AcquisitionFunctionName',acquisitionFcn,...   
  'SaveFileName',results_filename,...   
  'OutputFcn',@saveToFile);
diary off
%% 
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