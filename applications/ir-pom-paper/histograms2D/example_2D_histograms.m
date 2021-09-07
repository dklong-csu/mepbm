%% Data import
% The c++ code that generates the samples creates files named
% 'sample.N.txt' where N is the Nth chain that was run.
% For this paper, 232 chains of 20,000 samples were run for each, so there
% are files samples.0.txt, ... , samples.231.txt.
root_folder = '../mcmc/';
%data_3step_t1 = import_data(strcat(root_folder, 'three_step_time1/samples/samples.'),0:231);
%data_3step_t2 = import_data(strcat(root_folder, 'three_step_time2/samples/samples.'),0:231);
%data_3step_t3 = import_data(strcat(root_folder, 'three_step_time3/samples/samples.'),0:231);
%data_3step_t4 = import_data(strcat(root_folder, 'three_step_time4/samples/half_dt/samples.'),0:231);
data_3step_all = import_data(strcat(root_folder, 'three_step_all_times/samples/samples.'),0:231,1000);
data_4step_all = import_data(strcat(root_folder, 'four_step_all_times/samples/samples.'),0:231,1000);

%% 2D Histograms -- 3-step
vars = {'k_b','k_1','k_2','k_3','M'};
%vars = {'k_b','k_1'};

% The first column gives the same information as the second column
data = data_3step_all(:,2:end);

for i=1:length(vars)
    for j=i:length(vars)
        if i~=j
            f = create_kde_plot(data(:,i),data(:,j),30);
            file_name = strcat("3step_",vars{i},"_",vars{j},".png");
            exportgraphics(f, strcat('./',file_name),'Resolution',1200)
            close all
        end
    end
end

%% Make axis labels
for i=1:length(vars)
    fprintf("%s: min=%f, max=%f\n",vars{i},min(data(:,i)),max(data(:,i)))
end



%% 2D Histograms -- 4-step
vars = {'k_b','k_1','k_2','k_3','k_4','M'};
%vars = {'k_b','k_1','k_2'};

% The first column gives the same information as the second column
data = data_4step_all(:,2:end);

for i=1:length(vars)
    for j=i:length(vars)
        if i~=j
            f = create_kde_plot(data(:,i),data(:,j),30);
            file_name = strcat("4step_",vars{i},"_",vars{j},".png");
            exportgraphics(f, strcat('./',file_name),'Resolution',1200)
            close all
        end
    end
end


