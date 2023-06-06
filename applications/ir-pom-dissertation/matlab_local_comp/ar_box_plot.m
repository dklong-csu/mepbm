% close all
clc
clearvars

ar_3_1 = get_AR('results_dissertation_ready_1/Results_3step_step1_MH.mat');
ar_3_2 = get_AR('results_dissertation_ready_1/Results_3step_step2_AM.mat');
ar_3_3 = get_AR('results_dissertation_ready_1/Results_3step_step3_MH.mat');

ar_4_1 = get_AR('results_dissertation_ready_1/Results_4step_step1_MH.mat');
ar_4_2 = get_AR('results_dissertation_ready_1/Results_4step_step2_AM.mat');
ar_4_3 = get_AR('results_dissertation_ready_1/Results_4step_step3_MH.mat');

ar_BF_1 = get_AR('Results_Model_Select_Sym_ver_1.mat');
ar_BF_2 = get_AR('Results_Model_Select_Asym_ver_1.mat');

AR = [ar_3_1';ar_3_2';ar_3_3';ar_4_1';ar_4_2';ar_4_3';ar_BF_1';ar_BF_2'];

g1 = repmat({'3-step #1'},20,1);
g2 = repmat({'3-step #2'},20,1);
g3 = repmat({'3-step #3'},20,1);

g4 = repmat({'4-step #1'},20,1);
g5 = repmat({'4-step #2'},20,1);
g6 = repmat({'4-step #3'},20,1);

g7 = repmat({'Symmetric Model Select'},20,1);
g8 = repmat({'Asymmetric Model Select'},20,1);

G = [g1;g2;g3;g4;g5;g6;g7;g8];

figure('Position',[100 100 1300 1000])
boxchart(categorical(G),AR)
ylabel('Acceptance Ratio')
fh = findobj('Type','Figure');
set(findall(fh,'-property','FontSize'),'FontSize',20)
grid on


%% Functions
function result = extractAnalysis(file)
    load(file, "BI");
    result = BI;
end

function ar = get_AR(file)
    BI = extractAnalysis(file);
    ar = BI.Results.Acceptance;
end