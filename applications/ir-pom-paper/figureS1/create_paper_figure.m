%%
root_folder = '../mcmc/';
data = import_data(strcat(root_folder, 'three_step_all_times/samples/samples.'),0:231);

%%
n_rows = size(data,1);
data_subset = cell(5,1);
n = [1e5 5e5 1e6 2e6 size(data,1)];
for i=1:length(data_subset)
    data_subset{i} = data(1:n(i),4);
end

%%


for i=1:length(data_subset)
    num_samples = size(data_subset{i},1);
    plot_title = strcat(num2str(num_samples),' samples');
    plot_labels = {'','',plot_title};

    limits = [0, 6e5, 0, 1e-5];


    num_bins = 100;
    font_size = 20;


    % Set tick marks for the figures. The first is the x-axis ticks and the
    % second is the y-axis ticks. 'auto' lets Matlab decide the ticks and
    % '' makes no tick marks appear. 
    ticks = {'',''};
    tick_labels = { '', '' };

    f = figure;
    create_histogram1D(data_subset{i}, plot_labels, num_bins, ...
                       font_size, limits, ticks, tick_labels)

    % Export as a vector graphic since this is a relatively small
    % figure and a vector graphic ensures high resolution.
    %exportgraphics(f, strcat(folder,'/',file_names{i},'.pdf'),'ContentType','vector')
end
