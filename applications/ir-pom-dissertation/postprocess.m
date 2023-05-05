clearvars
clc
close all


addpath("/raid/long/p/uqlab/UQLab_Rel2.0.0/core")
uqlab -nosplash
myBI = extractAnalysis("Results_3step_MH_simplified_constants_ver_5.mat");

postProcess(myBI);


%%
function result = extractAnalysis(file)
    load(file, "BI");
    result = BI;
end




function postProcess(BI)
uq_postProcessInversion(BI, ...
    'gelmanRubin',true, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 1)
uq_print(BI)
uq_display(BI, ...,
    'scatterplot', 'none', ...
    'meanConvergence','none', ...
    'acceptance',true, ...
    'trace','all')

fileID = fopen("plot_MAP_prm.txt",'w');
fprintf(fileID, "%.40e\n", BI.Results.PostProc.PointEstimate.X{1});
fclose(fileID);

fileID = fopen("plot_mean_prm.txt",'w');
fprintf(fileID, "%.40e\n", BI.Results.PostProc.PointEstimate.X{2});
fclose(fileID);
end
