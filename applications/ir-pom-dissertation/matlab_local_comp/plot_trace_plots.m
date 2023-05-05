clearvars
clc
close all
warning('off')
%% Calculate rhat for each variable
uqlab -nosplash
myBI = extractAnalysis("Results_Model_Select_Sym_ver_1.mat");

uq_postProcessInversion(myBI, ...
    'gelmanRubin',true, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 2);

% model select
prm_names = ["k_f","k_b","k_1","k_2","k_3","M",...
    "k_f","k_b","k_1","k_2","k_3","k_4","M",...
    "w"];
prm_idx = [14];
% 3 step
% prm_names = ["k_f","k_b","k_1","k_2","k_3","M"];
% prm_idx = 1:6;
% 4 step
% prm_names = ["k_f","k_b","k_1","k_2","k_3","k_4","M"];
% prm_idx = 1:7;


%% Plot all samples
samps = myBI.Results.Sample;
[n, ~, c] = size(samps);
for ii=prm_idx
    figure('Position',[100 100 1300 1000])
    t = tiledlayout(1,2,'TileSpacing','tight','Padding','compact');
    plot_samps = zeros(n,c);
    plot_samps(:,:) = samps(:,ii,:);
    ax = nexttile(1);
    ax.FontSize = 20;
    box on
    grid on
    hold on
    for cc=1:c
        plot(1:n,plot_samps(:,cc)')
    end
    hold off
    ylabel(prm_names(ii),'Rotation',0,...
                    'HorizontalAlignment','right',...
                    'VerticalAlignment','middle')
    xlabel("Steps")

    yvals = ax.YLim;
    ax.Layer = 'top';
    yticks(linspace(yvals(1), yvals(2), 6))
    xlim([0, n])
    xticks(linspace(0,n,4))

    ax = nexttile(2);
    ax.FontSize = 20;
    box on
    grid on
    hold on
    for cc=1:c
        [f,xi] = ksdensity(plot_samps(:,cc));
        plot(f,xi)
    end
    [f,xi] = ksdensity(plot_samps(:));
    plot(f,xi,'Color','k','LineWidth',2)
    xlabel(strcat("\pi(",prm_names(ii),")"))
    
    hold off
    ax.Layer = 'top';
    yticks(linspace(yvals(1), yvals(2), 5))
    yticklabels({'','','','',''})
    ylim(yvals);

    xvals = ax.XLim;
    xticks(linspace(xvals(1), xvals(2),4))
end


%% Plot post processed chains
samps = myBI.Results.PostProc.PostSample;
[n, ~, c] = size(samps);
for ii=prm_idx
    figure('Position',[100 100 1300 1000])
    t = tiledlayout(1,2,'TileSpacing','tight','Padding','compact');
    pre_samps = zeros(n,c);
    pre_samps(:,:) = samps(:,ii,:);
    plot_samps = split_chains_in_half(pre_samps);
    ax = nexttile(1);
    ax.FontSize = 20;
    box on
    grid on
    hold on
    for cc=1:2*c
        plot(plot_samps(:,cc)')
    end
    hold off
    ylabel(prm_names(ii),'Rotation',0,...
                    'HorizontalAlignment','right',...
                    'VerticalAlignment','middle')
    xlabel("Steps")

    yvals = ax.YLim;
    ax.Layer = 'top';
    yticks(linspace(yvals(1), yvals(2), 6))

    xlim([0, n/2+1])
    xticks(linspace(0, n/2+1, 4))

    ax = nexttile(2);
    ax.FontSize = 20;
    box on
    grid on
    hold on
    for cc=1:2*c
        [f,xi] = ksdensity(plot_samps(:,cc));
        plot(f,xi)
    end
    [f,xi] = ksdensity(plot_samps(:));
    plot(f,xi,'Color','k','LineWidth',2)
    xlabel(strcat("\pi(",prm_names(ii),")"))
    
    hold off
    ax.Layer = 'top';
    yticks(linspace(yvals(1), yvals(2), 5))
    yticklabels({'','','','',''})
    ylim(yvals);

    xvals = ax.XLim;
    xticks(linspace(xvals(1), xvals(2),4))
end



%% functions
function result = extractAnalysis(file)
    load(file, "BI");
    result = BI;
end

function split = split_chains_in_half(samples)
    [n, m] = size(samples);
    split = zeros(n/2, m*2);
    for ii=1:m
        split(:, (ii-1)*2+1) = samples(1:n/2, ii);
        split(:, (ii-1)*2+2) = samples(n/2+1:end,ii);
    end
end