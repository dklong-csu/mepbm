% Call program to load TEM data
TEMdata

% Organize the data
data = {S2, S3, S4, S5};
times = [0.918 1.170 2.336 4.838];

% Information for saved figures
root_folder = './';
font_size = 18;

% Create figures and save down
for i=1:length(data)
   f = plot_tem_histogram(data{i}, [1.4 4.1], times(i), font_size);
   file_name = strcat('data_time', num2str(i));
   exportgraphics(f, ...
     strcat(root_folder,'/',file_name,'.pdf'), ...
     'ContentType',...
     'vector')
   close all
end

