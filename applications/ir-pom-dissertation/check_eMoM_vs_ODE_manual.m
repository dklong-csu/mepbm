clearvars
close all
clc

addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
uqlab -nosplash

%%
p = gcp('nocreate');
if isempty(p)
    parpool(96);
elseif p.NumWorkers < 96
    delete(gcp);
    parpool(96);
end

%%
load Results_eMoM_3step_MH_ver_8.mat
BI_eMoM = BI;
uq_postProcessInversion(BI_eMoM, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 2)
samps_eMoM = BI_eMoM.Results.PostProc.PostSample(:,:,:);

samps_check = samps_eMoM(1:end,:,1);
% s  = 11.3;
% kb = 0.8e4;
% kf = kb*5e-7;
% k1 = 1.5e5;
% k2 = 1.65e4;
% k3 = 5.63e3;
% M  = 100;
% samps_check = [kf, kb, k1, k2, k3, M];

%% 
tic
fprintf("Solving c++: ")
fileID = fopen('samples4cost.txt','w');
for ii=1:size(samps_check,1)
    for jj=1:size(samps_check,2)
        fprintf(fileID,"%f",samps_check(ii,jj));
        if jj < size(samps_check,2)
            fprintf(fileID,", ");
        end
    end
    fprintf(fileID,"\n");
end

%% ODE PSD calculation
[~,~] = system('./calc_concentrations_3step_simple');


%% Read particle concentrations in
n_particles = 2498; %2498
fileID = fopen('optimal_PSD.txt','r');
formatSpec = '%f';
sizeArray = [n_particles Inf];

PSDs = fscanf(fileID, formatSpec, sizeArray);

fclose(fileID);
toc
%% eMoM PSD calculation
tic
fprintf("Solving eMoM: ")
PSDs_eMoM = zeros(size(PSDs));
parfor ii=1:size(PSDs_eMoM,2)
    kf = samps_check(ii,1);
    kb = samps_check(ii,2);
    k1 = samps_check(ii,3);
    k2 = samps_check(ii,4);
    k3 = samps_check(ii,5);
    M = samps_check(ii,6);
    psd = MoM_PSD(kf,kb,k1,k2,k3,M,4.838)';
    PSDs_eMoM(:,ii) = psd;
end

toc
%%
diams = 0.3 * (3:2500).^(1/3);
figure('Position',[100 100 1300 1000])
hold on
for ii=1:size(PSDs,2)
    plot(diams', PSDs(:,ii),'k')
    plot(diams', PSDs_eMoM(:,ii),'r')
end
hold off