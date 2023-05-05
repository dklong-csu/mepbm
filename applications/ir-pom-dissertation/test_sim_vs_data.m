clearvars
clc
% close all

%% POM Data
TEMdata

obs_t1 = histcounts(S2, 1.4:.1:4.1);


%% Pretend to solve ODE a bunch of times
rng('default');
n_categories = 27;
probs = unifrnd(0,1,n_categories,1);
probs = probs / sum(probs);
pd = makedist('Multinomial',probs);


n_draws = sum(obs_t1);
n_samples = 100;
r = random(pd, n_draws,n_samples);

data = cell(1, n_categories);
for ii=1:n_categories
    counts = zeros(n_samples, 1);
    for jj=1:n_samples
        a = r(:,jj);
        n_cat = length(a(a==ii));
        counts(jj) = n_cat;
    end
    data{ii} = counts;
    
end

total_count = zeros(n_samples,1);
for ii=1:n_categories
    total_count = total_count + data{ii};
end

figure
xtick_skips = 2;
xtick_vals = 1:xtick_skips:27;
plot_labels = cell(1,length(xtick_vals));
edges = 1.4:.1:4.1;

for ii=1:length(xtick_vals)
    bin = xtick_vals(ii);
    plot_labels{ii} = strcat(num2str(edges(bin)),'-',num2str(edges(bin+1)),' nm');
end
violin(data,'facecolor',[0 0.4470 0.7410],'mc','','medc','','edgecolor','','facealpha',.6, 'support',[-0.1, 246],'Kernel','box');
ylabel('No. of observed particles')

%% Plot observed data within violin plot
hold on
for ii=1:length(obs_t1)
    scatter(ii, obs_t1(ii),'rx', 'LineWidth',1.5)
end
qw{1} = plot(nan,'rx');
qw{2} = plot(nan,'color',[0 0.4470 0.7410],'LineWidth',10);
legend([qw{:}],{'Observed','Simulated'})

xticks(xtick_vals);
xticklabels(plot_labels);
xtickangle(45)
%ylim([0 30])
hold off