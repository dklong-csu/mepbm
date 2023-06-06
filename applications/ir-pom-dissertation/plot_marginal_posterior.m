clearvars
clc
% close all
warning('off')
addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
%% Calculate rhat for each variable
uqlab -nosplash
myBI = extractAnalysis("Results_eMoM_3step_MH_ver_6.mat");

uq_postProcessInversion(myBI, ...
    'gelmanRubin',true, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 2);

% model select
% prm_name = "\omega";
% prm_idx = 14;
% 3 step
prm_names = ["k_f","k_b","k_1","k_2","k_3","M"];
prm_idx = 1:6;
% 4 step
% prm_names = ["k_f","k_b","k_1","k_2","k_3","k_4","M"];
% prm_idx = 1:7;

if length(prm_idx) == 1
    % plotting the mixture parameter
    samps = myBI.Results.PostProc.PostSample(:,:,:);
    [n, p, c] = size(samps);
    plot_samps = zeros(n,c);
    plot_samps(:,:) = samps(:,prm_idx,:);


    figure('Position',[100 100 1300 1000],'defaultAxesFontSize',18)
    hold on
    plot_colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980];
    colororder(plot_colors)

    % plot cdf
    yyaxis right
    [fc,xci] = ksdensity(plot_samps(:),'Function','cdf');
    area(xci, fc,'EdgeColor','none','DisplayName','CDF','FaceAlpha',.5,...
        'FaceColor',plot_colors(2,:));
    ylim([0 1])
    yticks(linspace(0,1,5))
    yticklabels(["0%","25%","50%","75%","100%"])
    ylabel("Cumulative Density",'FontSize',24)

    % plot pdf
    yyaxis left
    [f,xi] = ksdensity(plot_samps(:));
    area(xi,f,'EdgeColor','none','DisplayName','PDF','FaceAlpha',1,...
        'FaceColor',plot_colors(1,:));
    
    ylabel("Probability Density",'FontSize',24)
    yticks(linspace(gca().YLim(1), gca().YLim(2),5))

    % make things look nice
    grid on
    ax = gca;
    ax.Layer = 'top';
    ax.Box = 'on';
    xlim([0 1])
    xticks([0 .25 .5 .75 1])
    xlabel(prm_name,'FontSize', 24)
    title('Marginal Posterior Distribution','FontSize',24)

    

    % add a legend
    legend('Location','best')
    hold off
else
    samps = myBI.Results.PostProc.PostSample(:,:,:);
    [n, p, c] = size(samps);
    
    prm_min = zeros(1,p);
    prm_max = zeros(1,p);
    
    reds = linspace(1, 0, 100)';
    greens = linspace(1, .4470, 100)';
    blues = linspace(1, .7410, 100)';
    colorMap = [reds, greens, blues];
    
    figure('Position',[100 100 1300 1000])
    t = tiledlayout(length(prm_idx),length(prm_idx),'TileSpacing','compact','Padding','compact');
    pos = 1;
    for ii=prm_idx
        for jj=prm_idx
    %         pos = (ii-1)*p + jj;
            if ii==jj
                ax = nexttile(pos);
                % 1D marginal
                samps1 = zeros(n,c);
                samps1(:,:) = samps(:,ii,:);
                [f,xi] = ksdensity(samps1(:));
                prm_min(ii) = min(prm_min(ii), min(xi));
                prm_max(ii) = max(prm_max(ii), max(xi));
    %             plot(xi,f)
                area(xi,f,'EdgeColor','none')
                xticks('')
                yticks('')
                grid on
                ax.Layer = 'top';
                ax.Box = 'on';
            elseif ii>jj
                ax = nexttile(pos);
                % 2D marginal
                samps1 = zeros(n,c);
                samps1(:,:) = samps(:,jj,:);
                samps2 = zeros(n,c);
                samps2(:,:) = samps(:,ii,:);
                [f,xi] = ksdensity([samps1(:), samps2(:)]);
    
                prm_min(jj) = min(prm_min(jj), min(xi(:,1)));
                prm_max(jj) = max(prm_max(jj), max(xi(:,1)));
    
                prm_min(ii) = min(prm_min(ii), min(xi(:,2)));
                prm_min(ii) = max(prm_max(ii), max(xi(:,2)));
    
    
                [x,y,z] = computeGrid(xi(:,1), xi(:,2), f);
                surf(x,y,z,'EdgeColor','none','FaceColor','interp')
                xlim([min(xi(:,1)), max(xi(:,1))])
                ylim([min(xi(:,2)), max(xi(:,2))])
                view(2)
                colormap(colorMap)
                xticks('')
                yticks('')
                grid on
                ax.Layer = 'top';
                ax.Box = 'on';
            else
    %             nexttile
            end
            pos = pos + 1;
        end
    end
    
    pos = 1;
    for ii=prm_idx
        for jj=prm_idx
    %         pos = (ii-1)*p + jj;
            
            if ii==jj
                ax = nexttile(pos);
                L = ax.XLim;
                xspan = L(2) - L(1);
                offset = 0.15 * xspan;
                xlo = L(1) + offset;
                xhi = L(2) - offset;
                xticks([xlo xhi])
                xticklabels({'', ''})
            elseif ii>jj
                ax = nexttile(pos);
                xlim = [prm_min(jj) prm_max(jj)];
                ylim = [prm_min(ii), prm_max(ii)];
    
                L = ax.XLim;
                xspan = L(2) - L(1);
                offset = 0.15 * xspan;
                xlo = L(1) + offset;
                xhi = L(2) - offset;
                xticks([xlo xhi])
                xticklabels({'', ''})
    
                yspan = prm_max(ii) - prm_min(ii);
                offset = 0.15 * yspan;
                ylo = prm_min(ii) + offset;
                yhi = prm_max(ii) - offset;
                yticks([ylo yhi])
                yticklabels({'', ''})
            end
    
            if jj==1
                nexttile(pos)
                if prm_names(ii) == "M"
                    ylabel("M_ ",'FontSize',18,'Rotation',0,...
                    'HorizontalAlignment','right',...
                    'VerticalAlignment','middle')
                elseif ii == 1
                    ylabel("k_f      ",'FontSize',18,'Rotation',0,...
                    'HorizontalAlignment','right',...
                    'VerticalAlignment','middle')
                else
                    ylabel(prm_names(ii),'FontSize',18,'Rotation',0,...
                        'HorizontalAlignment','center',...
                        'VerticalAlignment','middle')
                end
            end
            if ii==p
                ax = nexttile(pos);
                xlabel(prm_names(jj),'FontSize', 18)
                L = ax.XLim;
                xspan = L(2) - L(1);
                offset = 0.15 * xspan;
                xlo = L(1) + offset;
                xhi = L(2) - offset;
                xticks([xlo xhi])
                xticklabels({num2scientificstr(xlo), num2scientificstr(xhi)})
            end
    
            if jj==1 && jj~=ii
                ax = nexttile(pos);
                L = ax.YLim;
                yspan = L(2) - L(1);
                offset = 0.15 * yspan;
                ylo = L(1) + offset;
                yhi = L(2) - offset;
                yticks([ylo yhi])
                yticklabels({num2scientificstr(ylo), num2scientificstr(yhi)})
            end
            pos = pos + 1;
        end
    end
    
    cb = colorbar('Ticks',[0.1, 0.9],...
        'TickLabels',{'Low probability','High probability'},...
        'FontSize', 18);
    cb.Layout.Tile = 'south';
    cb.TickLength = 0;
    
    title(t,'Marginal Posterior Distributions','FontSize',24)
end
%% functions

function result = extractAnalysis(file)
    load(file, "BI");
    result = BI;
end

function [xq,yq,z] = computeGrid(x1,x2,fout)
    x = linspace(min(x1),max(x1));
    y = linspace(min(x2),max(x2));
    [xq,yq] = meshgrid(x,y);
    orig_state = warning;
    warning('off','all');
    z = griddata(x1,x2,fout,xq,yq);
    warning(orig_state);
end

function mystr = num2scientificstr(num)
    if num >= 1 && num<1000
        mystr = num2str(num, "%.1f");
    else
        mag = floor(log10(num));
        leadingnum = num / (10^mag);
        mystr = strcat( num2str(leadingnum,"%.1f"), ...
            "\times10^{",...
            num2str(mag,"%d"),...
            "}");
    end
end




