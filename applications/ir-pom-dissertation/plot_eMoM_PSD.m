clearvars
close all
clc

%%
kb = 7.5e+03;
kf = 0.0012;
k1 = 5.9e+05;
k2 = 1.9e+06;
k3 = 5.6e+03;
M = 110;
t = [0.918, 1.17, 2.336,4.838];

%% C++ calc
fileID = fopen('prm_simple_ode.txt','w');
fprintf(fileID,'%f\n',[kf,kb,k1,k2,k3,M]);

fprintf("C++ solve: ")
tic()
system('./calc_concentrations_3step_simple prm_simple_ode.txt prm_simple_psd');
toc()

%% eMoM calc
fprintf("eMoM solve: ")
tic()
[ll, psd_emom] = MoM_likelihood(kb,kf,k1,k2,k3,M);
toc()

%% time 1
x = 0.3 * (3:2500).^(1/3);
conc_ODE = parse_numeric_output("prm_simple_psd-1.out");
ode_max = max(conc_ODE);
conc_ODE = max(conc_ODE,1e-9*ode_max)';
conc_emom = psd_emom{1};

figure('Position',[100, 600, 575, 425])
scatter(x,conc_emom,'DisplayName',"eMoM")
hold on
scatter(x,conc_ODE,'DisplayName',"ODE")
legend
title("t = 0.918")
hold off


%% time 2
x = 0.3 * (3:2500).^(1/3);
conc_ODE = parse_numeric_output("prm_simple_psd-2.out");
ode_max = max(conc_ODE);
conc_ODE = max(conc_ODE,1e-9*ode_max)';
conc_emom = psd_emom{2};

figure('Position',[700, 600, 575, 425])
scatter(x,conc_emom,'DisplayName',"eMoM")
hold on
scatter(x,conc_ODE,'DisplayName',"ODE")
legend
title("t = 1.170")
hold off

%% time 3
x = 0.3 * (3:2500).^(1/3);
conc_ODE = parse_numeric_output("prm_simple_psd-3.out");
ode_max = max(conc_ODE);
conc_ODE = max(conc_ODE,1e-9*ode_max)';
conc_emom = psd_emom{3};

figure('Position',[100, 75, 575, 425])
scatter(x,conc_emom,'DisplayName',"eMoM")
hold on
scatter(x,conc_ODE,'DisplayName',"ODE")
legend
title("t = 2.336")
hold off

%% time 4
x = 0.3 * (3:2500).^(1/3);
conc_ODE = parse_numeric_output("prm_simple_psd-4.out");
ode_max = max(conc_ODE);
conc_ODE = max(conc_ODE,1e-9*ode_max)';
conc_emom = psd_emom{4};

figure('Position',[700, 75, 575, 425])
scatter(x,conc_emom,'DisplayName',"eMoM")
hold on
scatter(x,conc_ODE,'DisplayName',"ODE")
legend
title("t = 4.838")
hold off