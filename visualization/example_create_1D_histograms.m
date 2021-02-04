% This is an example of how to use the matlab functions in this directory
% to create histograms comparing parameters between two different models.
% 
% In this example we are comparing the results of two models each of which
% have two parameters. The samples for the first model are included in the
% directory `./samples/example_model_1/` and the samples for the second 
% model are included in the directory `./samples/example_model_2/`. Each
% model has files `samples.1.txt` and `samples.2.txt` and the files are
% formatted 
%       number, number
%       number, number
%       ...
% where the first column corresponds to parameter 1 and the second column
% corresponds to parameter 2.
%
% The following demonstrates how to import the data, combine all samples
% from a particular model into a single list of all samples, separate the
% list of samples into lists for each parameter, and finally create
% histograms for each model-parameter combination.

%% Data import
% Model number 1
data_model1 = import_data('./samples/example_model_1/samples.',[1 2]);

% Model number 2
data_model2 = import_data('./samples/example_model_2/samples.',[1 2]);

%% Parameter 1 plot
prm1_labels = {'Parameter 1', 'Probability density', 'My title'};
prm1_num_bins = 3;
prm1_font_size = 16;
prm1_limits = [0, 5, -Inf, Inf]; % Specify x-axis bounds but have automatic y-axis bounds 
prm1_ticks = {'auto', 'auto'};  % Let Matlab decide what ticks to use


% Model 1
prm1_model1 = data_model1(:,1);
figure('Name','Parameter 1: Model 1')
create_histogram1D(prm1_model1, prm1_labels, prm1_num_bins, ...
                   prm1_font_size, prm1_limits, prm1_ticks) 
exportgraphics(gcf, './prm1_model1.pdf') % If you want to save the generated histogram as a pdf


% Model 2
prm1_model2 = data_model2(:,1);
figure('Name','Parameter 1: Model 2')
create_histogram1D(prm1_model2, prm1_labels, prm1_num_bins, ...
                   prm1_font_size, prm1_limits, prm1_ticks) 
exportgraphics(gcf, './prm1_model2.pdf')


%% Parameter 2 plot
prm2_labels = {'Parameter 2', 'Probability density', 'My title'};
prm2_num_bins = 3;
prm2_font_size = 16;
prm2_limits = [0, 5, -Inf, Inf]; % Specify x-axis bounds but have automatic y-axis bounds 
prm2_ticks = {'auto', 'auto'};  % Let Matlab decide what ticks to use


% Model 1
prm2_model1 = data_model1(:,2);
figure('Name','Parameter 2: Model 1')
create_histogram1D(prm2_model1, prm2_labels, prm2_num_bins, ...,
                   prm2_font_size, prm2_limits, prm2_ticks) 
exportgraphics(gcf, './prm2_model1.pdf')


% Model 2
prm2_model2 = data_model2(:,2);
figure('Name','Parameter 2: Model 2')
create_histogram1D(prm2_model2, prm2_labels, prm2_num_bins, ...
                   prm2_font_size, prm2_limits, prm2_ticks) 
exportgraphics(gcf, './prm2_model2.pdf')

