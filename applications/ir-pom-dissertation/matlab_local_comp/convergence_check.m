clearvars
clc
close all
warning('off')
%% Calculate rhat for each variable
uqlab -nosplash
myBI = extractAnalysis("Results_Model_Select_Asym_ver_1.mat");

uq_postProcessInversion(myBI, ...
    'gelmanRubin',true, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 2);
uq_print(myBI)

% model select
prm_names = ["w"];
prm_idx = [14];
% 3 step
% prm_names = ["k_f","k_b","k_1","k_2","k_3","M"];
% prm_idx = 1:6;
% 4 step
% prm_names = ["k_f","k_b","k_1","k_2","k_3","k_4","M"];
% prm_idx = 1:7;


for ii=1:length(prm_idx)
    [nsamps, ~, nchains] = size(myBI.Results.PostProc.PostSample(:,prm_idx(ii),:));
    samps1D = zeros(nsamps, nchains);
    samps1D(:,:) = myBI.Results.PostProc.PostSample(:,prm_idx(ii),:);
    
    samps_split = split_chains_in_half(samps1D);
    rhat = calc_rhat(samps_split);
    fprintf("%s rhat = %f\n", prm_names(ii), rhat)
end





%% function
function B = calc_B(samples)
    % size(samples) = [n_samples, n_chains]
    % samples only represents a single parameter
    % Notation: n=length of chain, m=number of chains
    [n, m] = size(samples);
    chain_avg = mean(samples);
    all_avg = mean(chain_avg);
    B = n/(m-1) * sum( (chain_avg - all_avg).^2 );
end

function W = calc_W(samples)
    % size(samples) = [n_samples, n_chains]
    % samples only represents a single parameter
    % Notation: n=length of chain, m=number of chains
    [~, m] = size(samples);
    chain_var = var(samples);
    W = sum(chain_var) / m;
end

function vhat = calc_vhat(samples)
    B = calc_B(samples);
    W = calc_W(samples);
    [n, ~] = size(samples);
    vhat = (n-1)/n * W + B/n;
end

function rhat = calc_rhat(samples)
    rhat = sqrt( calc_vhat(samples) / calc_W(samples));
end

function split = split_chains_in_half(samples)
    [n, m] = size(samples);
    split = zeros(n/2, m*2);
    for ii=1:m
        split(:, (ii-1)*2+1) = samples(1:n/2, ii);
        split(:, (ii-1)*2+2) = samples(n/2+1:end,ii);
    end
end

function result = extractAnalysis(file)
    load(file, "BI");
    result = BI;
end

