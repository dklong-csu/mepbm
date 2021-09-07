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
data_4step_all = import_data(strcat(root_folder, 'four_step_all_times/samples/samples.'),0:231,1000);


%% Create marginal distributions

% Create a container that holds some descriptors of each set of samples.
% The third entry of each descriptor is a subdirectory that the figures
% will be saved in. This subdirectory needs to be created before running
% the code since Matlab will not automatically create a directory that does
% not exist. 
% { model, times, folder_name }
scenarios = { {'4 step', 'all', 'all_times_4step'} ...
              {'3 step', 'all', 'all_times_3step'} ...
              {'3 step', 'time 1', 'time1_3step'} ...
              {'3 step', 'time 2', 'time2_3step'} ...
              {'3 step', 'time 3', 'time3_3step'} ...
              {'3 step', 'time 4', 'time4_3step'} };
              
% Loop over each scenario -- set of samples -- and process the data to
% create a figure
for s=1:length(scenarios)

    model = scenarios{s}{1};
    times = scenarios{s}{2};
    folder = scenarios{s}{3};

    switch model
        case '4 step'
            nvars = 7;
            data_model = data_4step_all;
            
            % The plotting function takes an argument describing axis
            % labels and title. The order is { x-axis, y-axis, title }.
            % For the 4-step plots, we decided to include the x-axis label
            % but not the y-axis or title. The y-axis and title are useful
            % when creating intermediate figures throughout the research
            % process prior to the final set of figures being created.
            kf_labels = {'', '' , 'k_f' };
            kb_labels = {'', '', 'k_b' };
            k1_labels = {'', '', 'k_1' };
            k2_labels = {'', '', 'k_2' };
            k3_labels = {'', '', 'k_3' };
            k4_labels = {'', '', 'k_4' };
            M_labels = {'', '', 'M' };

            plot_labels = {kf_labels, kb_labels, k1_labels, k2_labels, k3_labels, k4_labels, M_labels};
            file_names = {'k_f','k_b','k_1','k_2','k_3','k_4','M'};

            % Limits for the x- and y-axes. Each row is organized
            % x minimum, x maximum, y minimum, y maximum
            % A value of Inf tells Matlab to decide what the value should
            % be. Prior to the final set of figures, it is helpful to have
            % each row be [0, Inf, 0, Inf] and turning on the y-axis tick
            % marks to determine what bounds look good for all of the 
            % figures.
            % The bounds below were decided via trial and error to in order
            % to create "nice-looking" figures.
            limits = [0, 0.1, 0, 1100;
                      0, 2e5, 0, 4e-5;
                      0, 4e5, 0, 8e-5;
                      0, 4e4, 0, 4e-4;
                      0, 3e4, 0, 1.5e-3;
                      0, 6000, 0, 6e-4;
                      0, 200, 0, 0.08];
                  
            xticks = { 'auto', ...
                       [0, 1e5, 2e5], ...
                       [0, 2e5, 4e5], ...
                       [0, 2e4, 4e4], ...
                       [0, 1.5e4, 3e4], ...
                       [0, 3e3, 6e3], ...
                       [0, 100, 200] };
                   
            xtick_labels = { 'auto', ...
                             {'0', '1\times10^5','2\times10^5'}, ...
                             {'0', '2\times10^5','4\times10^5'}, ...
                             {'0', '2\times10^4','4\times10^4'}, ...
                             {'0', '1.5\times10^4','3\times10^4'}, ...
                             {'0', '3\times10^3','6\times10^3'}, ...
                             {'0', '1\times10^2','2\times10^2'} };
        case '3 step'
            nvars = 6;

            switch times
                case 'all'
                    data_model = data_3step_all;
                case 'time 1'
                    data_model = data_3step_t1;
                case 'time 2'
                    data_model = data_3step_t2;
                case 'time 3'
                    data_model = data_3step_t3;
                case 'time 4'
                    data_model = data_3step_t4;
                otherwise
                    warning('Unexpected data time. No plots created.')
            end

            % The plotting function takes an argument describing axis
            % labels and title. The order is { x-axis, y-axis, title }.
            % For the 3-step plots, we decided to not include any labels
            % since we indicate the x-axis label information clearly in
            % the paper and the y-axis and title are not particularly
            % meaningful. 
            % The y-axis and title are useful when creating intermediate 
            % figures throughout the research process prior to the final 
            % set of figures being created.
            kf_labels = {'', '' , '' };
            kb_labels = {'', '', '' };
            k1_labels = {'', '', '' };
            k2_labels = {'', '', '' };
            k3_labels = {'', '', '' };
            M_labels = {'', '', '' };

            plot_labels = {kf_labels, kb_labels, k1_labels, k2_labels, k3_labels, M_labels};
            file_names = {'k_f','k_b','k_1','k_2','k_3','M'};

            % Limits for the x- and y-axes. Each row is organized
            % x minimum, x maximum, y minimum, y maximum
            % A value of Inf tells Matlab to decide what the value should
            % be. Prior to the final set of figures, it is helpful to have
            % each row be [0, Inf, 0, Inf] and turning on the y-axis tick
            % marks to determine what bounds look good for all of the 
            % figures.
            % The bounds below were decided via trial and error to in order
            % to create "nice-looking" figures.
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

        otherwise
            error('Unexpected model. No plots created.')
    end


    num_bins = 100;
    % The parameter M is integer-valued so it makes sense to plot this data
    % a little differently than the real-valued parameters. In particular,
    % we noticed that if not specified, the bin widths would not be integer
    % values and this caused the visualizations to be artificially jagged.
    integer_bin_width = 1;
    % Set a font size for any text that might appear on the figure.
    font_size = 20;
    
    
    % The last value in each set of parameters is integer-valued so we want
    % to use different plotting logic
    for i=1:nvars-1
        % Set tick marks for the figures. The first is the x-axis ticks and the
        % second is the y-axis ticks. 'auto' lets Matlab decide the ticks and
        % '' makes no tick marks appear. 
        ticks = {xticks{i},''};
        tick_labels = { xtick_labels{i}, '' };
        
        f = figure;
        create_histogram1D(data_model(:,i), plot_labels{i}, num_bins, ...
                           font_size, limits(i,:), ticks, tick_labels)
                       
        % Export as a vector graphic since this is a relatively small
        % figure and a vector graphic ensures high resolution.
        exportgraphics(f, strcat('./',folder,'_',file_names{i},'.pdf'),'ContentType','vector')
    end

    % Create the plot for the integer-valued parameter M.
    
    % Set tick marks for the figures. The first is the x-axis ticks and the
    % second is the y-axis ticks. 'auto' lets Matlab decide the ticks and
    % '' makes no tick marks appear. 
    ticks = {xticks{end},''};
    tick_labels = { xtick_labels{end}, '' };
    
    f = figure; 
    create_histogram1D_integer(data_model(:,nvars), plot_labels{nvars}, integer_bin_width, ...
                           font_size, limits(nvars,:), ticks, tick_labels)
    exportgraphics(f, strcat('./',folder,'_',file_names{nvars},'.pdf'),'ContentType','vector')

    % Close all of the figures before moving to the next set of samples.
    close all
end



