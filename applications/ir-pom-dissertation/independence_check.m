clearvars
clc
close all
warning('off')
%% Calculate effective sample size and thin
uqlab -nosplash
myBI = extractAnalysis("results/normal_approx/Results_3step_MHv2_ver_4.mat");

uq_postProcessInversion(myBI, ...
    'gelmanRubin',true, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 2);
uq_print(myBI)

% model select
% prm_names = ["w"];
% prm_idx = [14];
% 3 step
prm_names = ["k_f","k_b","k_1","k_2","k_3","M"];
prm_idx = 1:6;
% 4 step
% prm_names = ["k_f","k_b","k_1","k_2","k_3","k_4","M"];
% prm_idx = 1:7;

raw_samps = myBI.Results.PostProc.PostSample(:,:,:);

[thin_samps, ESS, rho, iat] = thin_chains(raw_samps);

%%
fprintf("     ")
fprintf("%5s   ",prm_names);
fprintf("\n");
fprintf("--------------------------------------------------\n")
fprintf("ESS: ")
fprintf("%5d   ", floor(ESS))
fprintf("\n")
fprintf("IAT: ")
fprintf("%5.1f   ",iat)
fprintf("\n")

[n_prm,n_samp] = size(rho);
lag=0:n_samp-1;

figure('Position',[100 100 1300 1000])
hold on
for i=1:n_prm
%     scatter(lag,rho(i,:),'filled','DisplayName',prm_names(i))
    plot(lag,rho(i,:),'LineWidth',10,'DisplayName',prm_names(i))
end
legend('location','southwest','NumColumns',4)
xlabel('lag')
ylabel('autocorrelation')
plot(lag,0*lag,'--k','HandleVisibility','off')
title("3-Step Analysis C")
set(gca,'FontSize',42)
xlim([0 10000])
ylim([-1 1])
hold off

%%
function result = extractAnalysis(file)
    load(file, "BI");
    result = BI;
end