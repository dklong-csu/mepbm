clearvars
clc
close all
uqlab

combined_bi = extractAnalysis("BI_seed_1_ver_1.mat");

postProcess(combined_bi);


%%
function result = extractAnalysis(file)
    load(file, "BI");
    result = BI;
end




function postProcess(BI)
uq_postProcessInversion(BI, ...
    'gelmanRubin',true, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 0.5)
uq_print(BI)
uq_display(BI, ...,
    'scatterplot', 'all', ...
    'meanConvergence','all', ...
    'acceptance',true, ...
    'trace','all')

fileID = fopen("plot_MAP_prm.txt",'w');
fprintf(fileID, "%.40e\n", BI.Results.PostProc.PointEstimate.X{1});
fclose(fileID);

fileID = fopen("plot_mean_prm.txt",'w');
fprintf(fileID, "%.40e\n", BI.Results.PostProc.PointEstimate.X{2});
fclose(fileID);
end
