close all
clearvars
clc

%%
TEMdata

bin_edges = 1.4:0.1:4.1;

%% Raw data + sim
rng(1,'twister')
prob_data = diameters_to_bin_probability(S2, bin_edges);

% temp sim data
s=(3:2500)';
diams = 0.3000805 .* s .^ (1/3);
conc = 1e-5 * normpdf(diams, mean(S2), 0.5);
prob_sim = sim_concentration_to_bin_probability(conc, diams, bin_edges);

% average together data and simulation
n_obs = 100;
prob_avg = (prob_data + prob_sim)/2;
obs = draw_N_from_bins(prob_avg, n_obs);

% comparative plots
obs_data = draw_N_from_bins(prob_data, n_obs);
obs_sim = draw_N_from_bins(prob_sim, n_obs);

figure
hold on
histogram('BinEdges',bin_edges,'BinCounts',obs,'DisplayName','avg')
histogram('BinEdges',bin_edges,'BinCounts',obs_data,'DisplayName','data')
histogram('BinEdges',bin_edges,'BinCounts',obs_sim,'DisplayName','sim')
legend
hold off

%% KDE of raw data
rng(1,'twister')
prob_data = diameters_to_bin_probability(S2, bin_edges);

prob_kde = kde_to_bin_probability(S2, bin_edges);

n_obs = 100;
obs_kde = draw_N_from_bins(prob_kde, n_obs);

% comparative plots
obs_data = draw_N_from_bins(prob_data, n_obs);

figure
hold on
histogram('BinEdges',bin_edges,'BinCounts',obs_kde,'DisplayName','kde')
histogram('BinEdges',bin_edges,'BinCounts',obs_data,'DisplayName','data')
legend
hold off

%% KDE of raw data + sim
rng(1,'twister')
prob_data = diameters_to_bin_probability(S2, bin_edges);

prob_kde = kde_to_bin_probability(S2, bin_edges);

% temp sim data
s=(3:2500)';
diams = 0.3000805 .* s .^ (1/3);
conc = 1e-5 * normpdf(diams, mean(S2), 0.5);
prob_sim = sim_concentration_to_bin_probability(conc, diams, bin_edges);

% average together data and simulation
n_obs = 100;
prob_avg = (prob_kde + prob_sim)/2;
obs = draw_N_from_bins(prob_avg, n_obs);

% comparative plots
obs_data = draw_N_from_bins(prob_data, n_obs);
obs_sim = draw_N_from_bins(prob_sim, n_obs);

figure
hold on
histogram('BinEdges',bin_edges,'BinCounts',obs,'DisplayName','avg')
histogram('BinEdges',bin_edges,'BinCounts',obs_data,'DisplayName','kde')
histogram('BinEdges',bin_edges,'BinCounts',obs_sim,'DisplayName','sim')
legend
hold off


        
        
        
        