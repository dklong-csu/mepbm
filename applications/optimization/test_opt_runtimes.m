clearvars
clc
close all

%%
p = gcp('nocreate');
if isempty(p)
    parpool(96);
elseif p.NumWorkers < 96
    delete(gcp);
    parpool(96);
end


post_filename = "thinned_posterior_samples.mat";
totalSampsPerDataset = 1000;%1000

A0      = 0.0012;
POM0    = 0;
Solvent = 11.3;
Time0   = 0.918;
Time1   = 1.17;
Time2   = 2.3;
Time3   = 4.8;

% designVars = table(A0, POM0, Solvent, Time0, Time1, Time2, Time3);
designVars = table(Time0);

optFcn = @(expDesign) -calculateMutualInformation(expDesign,post_filename,totalSampsPerDataset);

for i=1:1
tic
rng default
optFcn(designVars)
toc
end

