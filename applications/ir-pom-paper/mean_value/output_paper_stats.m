%% Import data
root_folder = '../mcmc/';
data_3step_t4 = import_data(strcat(root_folder, 'three_step_time4/samples/half_dt/samples.'),0:231,1000);
data_3step_all = import_data(strcat(root_folder, 'three_step_all_times/samples/samples.'),0:231,1000);
data_4step_all = import_data(strcat(root_folder, 'four_step_all_times/samples/samples.'),0:231,1000);


%% 3-Step exports
vars = {'k_b','k_1','k_2','k_3','M'};

data = data_3step_t4(:,2:end);
fprintf('3-step time 4\n')
fprintf('---------------------------------------------------\n')
for i=1:size(data,2)
    print_formatted_stats(data(:,i),vars{i})
end
fprintf('---------------------------------------------------\n')


data = data_3step_all(:,2:end);
fprintf('3-step all\n')
fprintf('---------------------------------------------------\n')
for i=1:size(data,2)
    print_formatted_stats(data(:,i),vars{i})
end
fprintf('---------------------------------------------------\n')

%% 4-Step exports
vars = {'k_b','k_1','k_2','k_3','k_4','M'};

data = data_4step_all(:,2:end);
fprintf('4-step all\n')
fprintf('---------------------------------------------------\n')
for i=1:size(data,2)
    print_formatted_stats(data(:,i),vars{i})
end
fprintf('---------------------------------------------------\n')