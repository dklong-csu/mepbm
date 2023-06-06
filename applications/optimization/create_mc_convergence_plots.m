clearvars
clc
close all

diary convergence_check_plots.log

%%
% diary off
% tic
% system('./convergence_check 1000000');
% diary on
% toc

%%
font_size = 42;
%%
EMV = parse_numeric_output('moving_avg_EMV.txt');
EPV = parse_numeric_output('moving_avg_EPV.txt');
varMV = parse_numeric_output('moving_avg_varMV.txt');
varPV = parse_numeric_output('moving_avg_varPV.txt');
EAC = parse_numeric_output('moving_avg_EAC.txt');

%%
fprintf("1%% plots\n")
tol = 0.01;

tic
makeConvergencePlot(EMV,tol)
xlabel('Number of Monte Carlo samples')
ylabel('EMD')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
ax = gca;
ax.YAxis.Exponent = -2;
exportgraphics(gca,'incremental_cost_EMD_1perc.png');
toc
%%
makeConvergencePlot(EPV,tol)
xlabel('Number of Monte Carlo samples')
ylabel('EPS')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
ax = gca;
ax.YAxis.Exponent = -1;
exportgraphics(gca,'incremental_cost_EPV_1perc.png');

makeConvergencePlot(varMV,tol)
xlabel('Number of Monte Carlo samples')
ylabel('VMD')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
exportgraphics(gca,'incremental_cost_varMD_1perc.png');

makeConvergencePlot(varPV,tol)
xlabel('Number of Monte Carlo samples')
ylabel('VPS')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
exportgraphics(gca,'incremental_cost_varPV_1perc.png');

makeConvergencePlot(EAC,tol)
xlabel('Number of Monte Carlo samples')
ylabel('EAC')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
exportgraphics(gca,'incremental_cost_EAC_1perc.png');


%%
fprintf("5%% plots\n")
tol = 0.05;
makeConvergencePlot(EMV,tol)
xlabel('Number of Monte Carlo samples')
ylabel('EMD')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
ax = gca;
ax.YAxis.Exponent = -2;
exportgraphics(gca,'incremental_cost_EMD_5perc.png');


makeConvergencePlot(EPV,tol)
xlabel('Number of Monte Carlo samples')
ylabel('EPS')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
ax = gca;
ax.YAxis.Exponent = -1;
exportgraphics(gca,'incremental_cost_EPV_5perc.png');

makeConvergencePlot(varMV,tol)
xlabel('Number of Monte Carlo samples')
ylabel('VMD')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
exportgraphics(gca,'incremental_cost_varMD_5perc.png');

makeConvergencePlot(varPV,tol)
xlabel('Number of Monte Carlo samples')
ylabel('VPS')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
exportgraphics(gca,'incremental_cost_varPV_5perc.png');

makeConvergencePlot(EAC,tol)
xlabel('Number of Monte Carlo samples')
ylabel('EAC')
set(findall(gcf,'-property','FontSize'),'FontSize',font_size)
exportgraphics(gca,'incremental_cost_EAC_5perc.png');

diary off
%%
function makeConvergencePlot(data, tol)
    figure('Position',[100 100 1300 1000])
    
    N = length(data);
    plot_idx = [1:1000, 1001:10:N];
    plot_data = data(plot_idx);
    fprintf("Number of plotted elements reduced by a factor of %f\n",length(data)/length(plot_data))
    
    
    plot(plot_idx, plot_data)
    set(gca,'XScale','log')
    x=[1, N];
    ub = data(end)*(1+tol);
    lb = data(end)*(1-tol);
    hold on
    shade(x,lb*ones(size(x)),x,ub*ones(size(x)),'FillType',[1 2;2 1],'LineStyle','none','FillColor','k','FillAlpha',.1);
    hold off
    idx = find(data >= ub | data <= lb);
    if isempty(idx)
        fprintf("Within tolerance after 1 sample.\n")
    else
        fprintf("Within tolerance after %d samples.\n",idx(end)+1)
    end
end