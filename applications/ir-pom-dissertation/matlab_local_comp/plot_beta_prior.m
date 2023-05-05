clearvars
clc
close all

%% Symmetric
% Note that x=0 --> 4step
%           x=1 --> 3step
figure('Position',[100 100 1300 1000])
tiledlayout(1,2,'TileSpacing','tight','Padding','compact')
x = linspace(0,1,100);

% pdf plot
a = .5;
b = .5;
y = betapdf(x,a,b);

ax1 = nexttile;
plot(x,y,'LineWidth',2)
title(strcat("\alpha=",num2str(a),", \beta=",num2str(b)),'FontSize',24)
ax1.FontSize = 20;
ylim([0 6])
yticks(linspace(0,6,5))
xticks(linspace(0,1,5))

% cdf plot
hold on
yyaxis right
p = betacdf(x,a,b);
plot(x,p,'LineWidth',2)
ylim([0 1])
yticks(linspace(0,1,5))
yticklabels({'','','','',''})
hold off
yyaxis left
grid on

%% Asymmetric favoring 3step
% pdf plot
% a=.8, b=.1 means the cdf for x=0.5 is 0.0899
% that is: I'm assuming a 8.99% "chance" that the 4step is correct
a = .6;
b = .4;
y = betapdf(x,a,b);

ax1 = nexttile;
plot(x,y,'LineWidth',2)
title(strcat("\alpha=",num2str(a),", \beta=",num2str(b)),'FontSize',24)
ax1.FontSize = 20;
ylim([0 6])
yticks(linspace(0,6,5))
yticklabels({'','','','',''})
xticks(linspace(0,1,5))

% cdf plot
hold on
yyaxis right
p = betacdf(x,a,b);
plot(x,p,'LineWidth',2)
ylim([0 1])
yticks(linspace(0,1,5))
hold off
yyaxis left
grid on




