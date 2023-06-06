clearvars
close all
clc
%%
load opt_exp_4times
tspan = linspace(0.1,10,50)';
[Time0,Time1,Time2,Time3] = ndgrid(tspan,tspan,tspan,tspan);
Time0 = Time0(:);
Time1 = Time1(:);
Time2 = Time2(:);
Time3 = Time3(:);
T = table(Time0,Time1,Time2,Time3);
[obj,sigma] = predictObjective(BayesoptResults,T);
%%
best = min(obj);
EI = 0*obj;
for i=1:length(EI)
    EI(i) = (best - obj(i))*normcdf(best,obj(i),sigma(i)) + sigma(i)*normpdf(best,obj(i),sigma(i));
end
max(EI)
min(EI)
mean(EI)
%%
figure
scatter(Time0,obj,'filled');
hold on
% scatter(Time1,obj);
% scatter(Time2,obj);
% scatter(Time3,obj);
hold off
legend