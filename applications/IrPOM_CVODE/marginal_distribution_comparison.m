%% Import new data
base_name_new = "/home/danny/r/mepbm/IrPomSundialsTest/mh_unif/samples.";
samples_per_chain = 20000;
chains_new=0:95;
n_chains = length(chains_new);
new_samples = zeros(samples_per_chain*n_chains,5);

for i=chains_new
    file_name = strcat(base_name_new,num2str(i),".txt");
    fileID = fopen(file_name,'r');
    formatSpec = '%f %f %f %f %f';
    sizeS = [5 Inf];
    
    chain = fscanf(fileID,formatSpec, sizeS);
    fclose(fileID);
    chain = chain';
    new_samples(1+i*samples_per_chain:(i+1)*samples_per_chain,:) = chain;
end

mean_new = mean(new_samples);
fprintf("Mean value of new samples: %f %f %f %f %f \n",mean_new)

%% Import new data - adaptive (chain 0 is not done yet)
base_name_amh = "/home/danny/r/mepbm/IrPomSundialsTest/amh_samples/samples.";
samples_per_chain = 20000;
chains_new_amh=0:95;
n_chains = length(chains_new_amh);
new_samples_amh = zeros(samples_per_chain*n_chains,5);

for i=chains_new_amh
    file_name = strcat(base_name_amh,num2str(i),".txt");
    fileID = fopen(file_name,'r');
    formatSpec = '%f %f %f %f %f';
    sizeS = [5 Inf];
    
    chain = fscanf(fileID,formatSpec, sizeS);
    fclose(fileID);
    chain = chain';
    new_samples_amh(1+i*samples_per_chain:(i+1)*samples_per_chain,:) = chain;
end

mean_new_amh = mean(new_samples_amh);
fprintf("Mean value of new samples adaptive: %f %f %f %f %f \n",mean_new_amh)


%% Import new data - adaptive with shared covariance (chain 0 is not done yet)
base_name_amh_shared = "/home/danny/r/mepbm/IrPomSundialsTest/amh_shared_samples/samples.";
samples_per_chain = 20000;
chains_new_amh_shared=0:95;
n_chains = length(chains_new_amh_shared);
new_samples_amh_shared = zeros(samples_per_chain*n_chains,5);

for i=chains_new_amh_shared
    file_name = strcat(base_name_amh_shared,num2str(i),".txt");
    fileID = fopen(file_name,'r');
    formatSpec = '%f %f %f %f %f';
    sizeS = [5 Inf];
    
    chain = fscanf(fileID,formatSpec, sizeS);
    fclose(fileID);
    chain = chain';
    new_samples_amh_shared(1+i*samples_per_chain:(i+1)*samples_per_chain,:) = chain;
end

mean_new_amh_shared = mean(new_samples_amh_shared);
fprintf("Mean value of new samples adaptive shared covariance: %f %f %f %f %f \n",mean_new_amh_shared)


%% Import old data -- 0-95
base_name_old = "/home/danny/r/ir_pom_paper_data/mcmc/three_step_all_times/samples/samples.";
samples_per_chain=20000;
chains_old_partial=0:95;
n_chains = length(chains_old_partial);
old_samples_partial = zeros(samples_per_chain*n_chains,6);

for i=chains_old_partial
    file_name = strcat(base_name_old,num2str(i),".txt");
    fileID = fopen(file_name,'r');
    formatSpec = '%f, %f, %f, %f, %f, %f';
    sizeS = [6 Inf];
    
    chain = fscanf(fileID, formatSpec, sizeS);
    fclose(fileID);
    chain = chain';
    old_samples_partial(1+i*samples_per_chain:(i+1)*samples_per_chain,:) = chain;
end
old_samples_partial = old_samples_partial(:,2:end);

mean_old_partial = mean(old_samples_partial);
fprintf("Mean value of old samples 0-95: %f %f %f %f %f \n",mean_old_partial)

%% Import old data -- 0-231
base_name_old = "/home/danny/r/ir_pom_paper_data/mcmc/three_step_all_times/samples/samples.";
samples_per_chain=20000;
chains_old_all=0:231;
n_chains = length(chains_old_all);
old_samples_all = zeros(samples_per_chain*n_chains,6);

for i=chains_old_all
    file_name = strcat(base_name_old,num2str(i),".txt");
    fileID = fopen(file_name,'r');
    formatSpec = '%f, %f, %f, %f, %f, %f';
    sizeS = [6 Inf];
    
    chain = fscanf(fileID, formatSpec, sizeS);
    fclose(fileID);
    chain = chain';
    old_samples_all(1+i*samples_per_chain:(i+1)*samples_per_chain,:) = chain;
end
old_samples_all = old_samples_all(:,2:end);

mean_old_all = mean(old_samples_all);
fprintf("Mean value of old samples 0-231: %f %f %f %f %f \n",mean_old_all)

%% Plot marginal distributions
kb_labels = {'', '', '' };
k1_labels = {'', '', '' };
k2_labels = {'', '', '' };
k3_labels = {'', '', '' };
M_labels = {'', '', '' };

plot_labels = {kb_labels, k1_labels, k2_labels, k3_labels, M_labels};
file_names = {'k_b','k_1','k_2','k_3','M'};

limits = [0, Inf, 0, Inf;
            0, Inf, 0, Inf;
            0, Inf, 0, Inf;
            0, Inf, 0, Inf;
            0, Inf, 0, Inf];
xticks = { 'auto', ...
            'auto', ...
            'auto', ...
            'auto', ...
            'auto' };
xtick_labels = { 'auto', ...
                 'auto', ...
                 'auto', ...
                 'auto', ...
                 'auto' };
             
num_bins = 100;
integer_bin_width = 1;
font_size = 20;

for i=1:4
    ticks = {xticks{i},''};
    tick_labels = {xtick_labels{i},''};
    
    figure;
    hold on
    create_histogram1D(new_samples(:,i), plot_labels{i}, num_bins, ...
                       font_size, limits(i,:), ticks, tick_labels)
    create_histogram1D(new_samples_amh(:,i), plot_labels{i}, num_bins, ...
                       font_size, limits(i,:), ticks, tick_labels)
    create_histogram1D(new_samples_amh_shared(:,i), plot_labels{i}, num_bins, ...
                       font_size, limits(i,:), ticks, tick_labels)
    create_histogram1D(old_samples_partial(:,i), plot_labels{i}, num_bins, ...
                       font_size, limits(i,:), ticks, tick_labels)
    create_histogram1D(old_samples_all(:,i), plot_labels{i}, num_bins, ...
                       font_size, limits(i,:), ticks, tick_labels)
    title(file_names{i})
    legend('SUNDIALS','SUNDIALS + Adaptive',...
           'SUNDIALS + Adaptive + Shared Covariance',...
           'Published 0-95','Published all')
    hold off
end

ticks = {xticks{end},''};
tick_labels = {xtick_labels{end},''};

figure;
hold on
create_histogram1D_integer(new_samples(:,end), plot_labels{5}, integer_bin_width, ...
                           font_size, limits(end,:), ticks, tick_labels)

create_histogram1D_integer(new_samples_amh(:,end), plot_labels{5}, integer_bin_width, ...
                           font_size, limits(end,:), ticks, tick_labels)
                       
create_histogram1D_integer(new_samples_amh_shared(:,end), plot_labels{5}, integer_bin_width, ...
                           font_size, limits(end,:), ticks, tick_labels)
                       
create_histogram1D_integer(old_samples_partial(:,end), plot_labels{5}, integer_bin_width, ...
                           font_size, limits(end,:), ticks, tick_labels)

create_histogram1D_integer(old_samples_all(:,end), plot_labels{5}, integer_bin_width, ...
                           font_size, limits(end,:), ticks, tick_labels)                       
title(file_names{end})
legend('SUNDIALS','SUNDIALS + Adaptive',...
       'SUNDIALS + Adaptive + Shared Covariance',...
       'Published 0-95','Published all')
hold off

