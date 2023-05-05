clearvars
clc
close all

diary ConvergenceMI.log
%%
p = gcp('nocreate');
if isempty(p)
    parpool(96);
elseif p.NumWorkers < 96
    delete(gcp);
    parpool(96);
end


posteriorFileName = "thinned_posterior_samples.mat";
totalSampsPerDataset = 100000;%1000



A0      = 0.0012;
POM0    = 0;
Solvent = 11.3;
Time0   = 0.918;
design = [A0, POM0, Solvent, Time0];

numerFcn = @(design, n_samps) simulateData(design, posteriorFileName, totalSampsPerDataset, totalSampsPerDataset+1);
denomFcn = @(design) cell2mat(simulateData(design, posteriorFileName, totalSampsPerDataset, 1));

lfireOpt.generateNumeratorData = numerFcn;
lfireOpt.generateDenominatorData = denomFcn;
lfireOpt.nMonteCarloSamples = totalSampsPerDataset;

rng default
tic
lr = LFIRE(design, lfireOpt);
toc
mi = 0*lr;
for i=1:length(mi)
    mi(i) = mean(lr(1:i));
end
figure('Position',[100 100 1500 1000])
plot(mi)
savefig('ConvergenceMI.fig')
diary off