clearvars
clc
close all


addpath("/raid/long/p/uqlab/UQLab_Rel2.0.0/core")
uqlab -nosplash
load("results/normal_approx/Results_3step_MHv2_ver_4.mat");

%% Setup
rng(1,'twister');
n_posterior_draws = 10000;

root_file_name = 'sim3step';
sim_exec_name = './calc_bin_probs_3step';

n_bins = 27;

%% Draw from posterior
uq_postProcessInversion(BI, ...
    'gelmanRubin',true, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 1)
fprintf("Drawing from the posterior...");
tstart = tic;
samps_chain = BI.Results.PostProc.PostSample;
[n, p, c] = size(samps_chain);

samps = zeros(n*c, p);
for ii=1:c
    samps(((ii-1)*n+1):ii*n,:) = samps_chain(:,:,ii);
end

% Calculate bandwidth based on Silverman's rule of thumb
sigma = std(samps);
bw = sigma .* (4/((p+2)*n*c)).^(1/(p+4));

% Randomly select samples
idx = randsample(n*c, n_posterior_draws);
posterior_draws = samps(idx, :);

% Perturb samples to simulate drawing from KDE
pert = mvnrnd(zeros(1,p), bw.^2, n_posterior_draws);
kde_draws = posterior_draws + pert;
kde_draws = max(kde_draws,1e-9); % for numerical stability
kde_draws(:,end) = max(3, kde_draws(:,end));

% Output to files for c++
for ii=1:n_posterior_draws
    fname = strcat(root_file_name, "-", num2str(ii-1), ".inp");
    fileID = fopen(fname,'w');
    s = kde_draws(ii,:);
    fprintf(fileID, "%.20f\n",s);
    fclose(fileID);
end
tstop = toc(tstart);
fprintf("done! (%.5f seconds)\n",tstop);
%% Call c++ to calculate bin probabilities
% Call c++
fprintf("Running c++ code...");
tstart = tic;
system_call = strcat(sim_exec_name, " ", root_file_name, " ",num2str(n_posterior_draws));
[status, cmdout] = system(system_call);

% Once c++ is run, extract bin probabilities from the generated files
sim_probs_time = cell(1,4);
for tt=1:4
    sim_probs = zeros(n_bins, n_posterior_draws);
    for ii=1:n_posterior_draws
        output_file = strcat(root_file_name, "-", num2str(ii-1),"-time",num2str(tt-1), ".out");
        sim_unnorm = max(0,parse_numeric_output(output_file));
        sim_unnorm_round = round(sim_unnorm, 20);
        sim_norm = sim_unnorm_round / sum(sim_unnorm_round);
        sim_norm = sim_norm / sum(sim_norm);
        sim_probs(:,ii) = sim_norm;
    end
    sim_probs_time{tt} = sim_probs;
end
tstop = toc(tstart);
fprintf("done! (%.5f seconds)\n",tstop);
%% Measured data
TEMdata

observed_time = {S2, S3, S4, S5};

%% Draw from multinomial distribution
fprintf("Drawing simulated data...");
tstart = tic;
sim_counts_time = cell(1,4);
% Loop over each time
for tt=1:4
    % measurement
    measured = histcounts(observed_time{tt}, 1.4:.1:4.1);
    n_particles = sum(measured);
    
    % Loop over each posterior draw
    r = zeros(n_particles, n_posterior_draws);
    for ii=1:n_posterior_draws
        % Draw from multinomial distribution
        probs = sim_probs_time{tt}(:,ii);
        %fprintf("sum probs = %.40f\n",sum(probs));
        pd = makedist('Multinomial',probs);
        r(:,ii) = random(pd,n_particles,1);
    end
    
    data = cell(1, n_bins);
    for ii=1:n_bins
        counts = zeros(n_posterior_draws, 1);
        for jj=1:n_posterior_draws
            a = r(:,jj);
            n_cat = length(a(a==ii));
            counts(jj) = n_cat;
        end
        data{ii} = counts;
    end
    sim_counts_time{tt} = data;
end
tstop = toc(tstart);
fprintf("done! (%.5f seconds)\n", tstop);
%% Create plots
fprintf("Plotting...");
tstart = tic;
xtick_skips = 4;
xtick_vals = 1:xtick_skips:n_bins;
plot_labels = cell(1,length(xtick_vals));
edges = 1.4:.1:4.1;

for ii=1:length(xtick_vals)
    bin = xtick_vals(ii);
    plot_labels{ii} = strcat(num2str(edges(bin),"%.1f"),'-',num2str(edges(bin+1),"%.1f"),' nm');
end

times = ["0.918", "1.170", "2.336", "4.838"];
for tt=1:4
    figure('Position',[100 100 1300 1000])
    % measurement
    measured = histcounts(observed_time{tt}, 1.4:.1:4.1);
    % plot simulation uncertainty
    violin(sim_counts_time{tt}, ...
        'facecolor',[0 0.4470 0.7410], ...
        'mc','', ...
        'medc','', ...
        'edgecolor','', ...
        'facealpha',.6, ...
        'support',[-0.1, sum(measured)], ...
        'Kernel','normal');
    ylabel('Number of observed particles')
    
    hold on
    for ii=1:length(measured)
        scatter(ii, measured(ii),60,"black","filled")
    end
    qw{1} = scatter(nan,nan,60,"black","filled");
    qw{2} = plot(nan,'color',[0 0.4470 0.7410],'LineWidth',10);
    legend([qw{:}],{'Observed','Simulated'})

    xticks(xtick_vals);
    xticklabels(plot_labels);
    xtickangle(45)
    ylim([0 1.5*max(measured)])
    title(strcat("3-step: ",times(tt)," hours"))
    grid on
    hold off
    fh = findobj('Type','Figure');
    set(findall(fh,'-property','FontSize'),'FontSize',24)
end
tstop = toc(tstart);
fprintf("done! (%.5f seconds)\n",tstop);

%% Delete generated files
delete(strcat(root_file_name,"*"))