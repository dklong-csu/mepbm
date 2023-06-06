clear vars
clc
close all
%% run 'check_eMoM_vs_ODE_manual.m' first
load psds_eMoM_BI.mat
TEMdata
%%
emom = PSDs_eMoM(:,1);
odes = PSDs(:,1);
x = 0.3 * (3:2500) .^(1/3);
bins = 1.4:0.1:4.1;
%%
figure('Position',[100 100 1300 1000])
emom_probs = sim_concentration_to_bin_probability(emom,x,bins);
odes_probs = sim_concentration_to_bin_probability(odes,x,bins);

histogram('BinEdges',bins,'BinCounts',emom_probs)
hold on
histogram('BinEdges',bins,'BinCounts',odes_probs)

set(gca, 'YScale','log')
% ylabel('log probability')
ylabel('probability')
xlabel('particle diameter (0.1 nm bins)')

yyaxis right
histogram(S5,bins)
ylabel('Data counts')
legend('eMoM','ODEs','Data counts')
title('eMoM vs ODEs -- Time 4.838 h')

hold off

%%
lle = calc_ll(S5,emom);
llo = calc_ll(S5,odes);
fprintf("eMoM log-likelihood: %f\n",lle)
fprintf("ODEs log-likelihood: %f\n",llo)

[dataE, llvecE] = calc_ll_unrolled(S5,emom);
[dataO, llvecO] = calc_ll_unrolled(S5,odes);
fprintf("eMoM | ODEs \n")
for ii=1:length(llvecE)
    fprintf("Bin %d:   ",ii)
    fprintf("%d * %f",dataE(ii),log(llvecE(ii)))
    fprintf(" | ")
    fprintf("%d * %f",dataO(ii),log(llvecO(ii)))
    fprintf(" | ")
    if dataE(ii) == 0
        fprintf("0%%")
    else
        pd = (dataE(ii)*log(llvecE(ii)) - dataO(ii)*log(llvecO(ii)) ) / (lle - llo);
        fprintf("%f%%",100*pd);
    end
    fprintf("\n")
end


%%
function ll = calc_ll(raw_data,psd)
    x = 0.3 * (3:2500) .^(1/3);
    bin_probs = sim_concentration_to_bin_probability(psd,x,1.4:0.1:4.1);
    bin_data = histcounts(raw_data,1.4:0.1:4.1);
    ll = dot(bin_data, log(bin_probs));
end

function [bin_data, bin_probs] = calc_ll_unrolled(raw_data,psd)
    x = 0.3 * (3:2500) .^(1/3);
    bin_probs = sim_concentration_to_bin_probability(psd,x,1.4:0.1:4.1);
    bin_data = histcounts(raw_data,1.4:0.1:4.1);
end