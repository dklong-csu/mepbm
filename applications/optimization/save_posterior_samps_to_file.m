clc
clearvars
close all

addpath('/home/nonfac/f/long/p/uqlab/UQLab_Rel2.0.0/core')
addpath('/home/nonfac/f/long/p/mepbm/lib/matlab')
uqlab -nosplash

% Reproducibility
rng('default')

%% Choose the posterior distribution
load /home/nonfac/f/long/p/mepbm/applications/ir-pom-dissertation/results/normal_approx/Results_3step_MHv2_ver_4.mat


uq_postProcessInversion(BI,'burnIn',2,...
    'pointEstimate',{'MAP','Mean'},...
    'gelmanRubin','true')

%% Thin chains to improve convergence of Monte Carlo integrals
tic
fprintf("Thinning chains...")
samps = BI.Results.PostProc.PostSample;

thinned_samps = thin_chains(samps);
split_samps = num2cell(thinned_samps,[1 2]);
stacked_samps = vertcat(split_samps{:});
save thinned_posterior_samples.mat stacked_samps
fprintf("done! ")
toc
%% Calculate bandwidth for kernel density estimation by Silverman's rule of thumb
tic
fprintf("Computing bandwidth...")
std_dev = std(stacked_samps);
bw = zeros(length(std_dev));
for ii=1:size(bw,1)
    d = size(bw,1);
    n = size(stacked_samps,1);
    bw(ii,ii) = ( 4 / (d+2) )^(1/ (d+4) ) *...
        n^(-1/(d+4)) *...
        std_dev(ii);
    bw(ii,ii) = bw(ii,ii)^2;
end
fprintf("done! ")
toc
%% Sample from the posterior
rng('default')
tic
fprintf("Sampling from the KDE...")
n_samps = 1000000;
% randomly select a row from the samples matrix
rows = randi(size(stacked_samps,1),n_samps,1);
% random perturbations based on bandwidth
pert = mvnrnd(zeros(1,size(stacked_samps,2)),bw,n_samps);

draws = 0*pert;
for ii=1:n_samps
    draws(ii,:) = stacked_samps(rows(ii),:) + pert(ii);
    draws(ii,1:end-1) = max(0, draws(ii,1:end-1));
    draws(ii,end) = max(3, draws(ii,end));
end
fprintf("done! ")
toc

%% Print samples to file
tic
fprintf("Writing samples to file...")
out_file = 'samples4cost.txt';
fileID = fopen(out_file,'w');
for ii=1:n_samps
    for jj=1:size(draws,2)
        fprintf(fileID,"%.20f",draws(ii,jj));
        if jj<size(draws,2)
            fprintf(fileID,", ");
        end
    end
    fprintf(fileID,"\n");
end

fclose(fileID);
fprintf("done! ")
toc