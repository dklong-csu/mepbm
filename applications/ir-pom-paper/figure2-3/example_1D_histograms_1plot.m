% This is the code used to generate Figure 2 and Figure 3. The created 
% samples are not included on GitHub, but this code is intended to be used 
% after the code located in all of the subdirectories of ../mcmc/ have been
% run and the resulting samples are moved to ../mcmc/<subdir>/samples/.

% This script imports the samples -- i.e. sets of parameters -- and plots
% the marginal distributions for each parameter for each set of samples.
% The marginal distributions are then exported as files that were inserted
% into the paper.

%% Data import
% The c++ code that generates the samples creates files named
% 'sample.N.txt' where N is the Nth chain that was run.
% For this paper, 232 chains of 20,000 samples were run for each, so there
% are files samples.0.txt, ... , samples.231.txt.
root_folder = '../mcmc/';
data_3step_t1 = import_data(strcat(root_folder, 'three_step_time1/samples/samples.'),0:231,1000);
data_3step_t2 = import_data(strcat(root_folder, 'three_step_time2/samples/samples.'),0:231,1000);
data_3step_t3 = import_data(strcat(root_folder, 'three_step_time3/samples/samples.'),0:231,1000);
data_3step_t4 = import_data(strcat(root_folder, 'three_step_time4/samples/half_dt/samples.'),0:231,1000);
data_3step_all = import_data(strcat(root_folder, 'three_step_all_times/samples/samples.'),0:231,1000);



%% Create marginal distributions

data = {data_3step_t1, data_3step_t2, data_3step_t3, data_3step_t4, data_3step_all};
nvars = 5;

kf_labels = {'', '' , '' };
kb_labels = {'', '', '' };
k1_labels = {'', '', '' };
k2_labels = {'', '', '' };
k3_labels = {'', '', '' };
M_labels = {'', '', '' };

plot_labels = {kf_labels, kb_labels, k1_labels, k2_labels, k3_labels, M_labels};
file_names = {'k_f','k_b','k_1','k_2','k_3','M'};

limits = [0, 0.03, 0, 1100;
          0, 4e4, 0, 6e-4;
          0, 4e5, 0, 3.5e-5;
          0, 6e5, 0, 4e-5;
          0, 3e4, 0, 1.5e-3;
          0, 200, 0, 0.08];
xticks = { 'auto', ...
           [0, 2e4, 4e4], ...
           [0, 2e5, 4e5], ...
           [0, 3e5, 6e5], ...
           [0, 1.5e4, 3e4], ...
           [0, 100, 200] };
xtick_labels = { 'auto', ...
                 {'0', '2\times10^4','4\times10^4'}, ...
                 {'0', '2\times10^5','4\times10^5'}, ...
                 {'0', '3\times10^5','6\times10^5'}, ...
                 {'0', '1.5\times10^4','3\times10^4'}, ...
                 {'0', '1\times10^2','2\times10^2'} };
             
num_bins = 100;
integer_bin_width = 1;
font_size = 20;

for n=1:nvars
    col = n+1;
    ticks = {xticks{col},''};
    tick_labels = { xtick_labels{col}, '' };
    
    f = figure;
    hold on
    for i=1:length(data)
        if n<nvars
            create_histogram1D(data{i}(:,col), plot_labels{col}, num_bins, ...
                               font_size, limits(col,:), ticks, tick_labels)
        else
            create_histogram1D_integer(data{i}(:,col), plot_labels{col}, integer_bin_width, ...
                                       font_size, limits(col,:), ticks, tick_labels)
        end
    end
    title(file_names{col})
    legend({'t_1','t_2','t_3','t_4','all'},'FontSize',font_size)
    hold off
    exportgraphics(f, strcat('./one_plot_',file_names{col},'.pdf'),'ContentType','vector')
end

close all

