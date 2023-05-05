clearvars
close all
clc

addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
uqlab -nosplash

%%
load Results_eMoM_3step_MH_ver_8.mat
BI_eMoM = BI;
uq_postProcessInversion(BI_eMoM, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 2)
mu_eMoM = BI_eMoM.Results.PostProc.PointEstimate.X{2};
cov_eMoM = BI_eMoM.Results.PostProc.Dependence.Cov;

load("Results_3step_MH_simplified_constants_ver_6.mat")
BI_ODE = BI;
uq_postProcessInversion(BI_ODE, ...
    'pointEstimate', {'MAP','Mean'}, ...
    'burnIn', 2)
mu_ODE = BI_ODE.Results.PostProc.PointEstimate.X{2};
cov_ODE = BI_ODE.Results.PostProc.Dependence.Cov;

%%
mu_diff = (mu_eMoM - mu_ODE) ./ mu_ODE;
cov_diff = (cov_eMoM - cov_ODE) ./ cov_ODE;

%%
% fprintf("Max mean diff = %f\n",max(mu_diff))
% fprintf("Max var diff = %f\n",max(max(cov_diff)));

%%
warning('off')
addpath('/raid/long/p/uqlab/UQLab_Rel2.0.0/core')
%% 


% 3 step
prm_names = ["k_f","k_b","k_1","k_2","k_3","M"];
prm_idx = 1:6;


samps = BI_ODE.Results.PostProc.PostSample(:,:,:);
samps_eMoM = BI_eMoM.Results.PostProc.PostSample(:,:,:);
[n, p, c] = size(samps);

prm_min = zeros(1,p);
prm_max = zeros(1,p);

reds = linspace(1, 0, 100)';
greens = linspace(1, .4470, 100)';
blues = linspace(1, .7410, 100)';
colorMap = [reds, greens, blues];

reds2 = linspace(1, 0.8500,100)';
greens2 = linspace(1, 0.3250, 100)';
blues2 = linspace(1, 0.0980, 100)';
colorMap2 = [reds2, greens2, blues2];

figure('Position',[100 100 1300 1000])
t = tiledlayout(length(prm_idx),length(prm_idx),'TileSpacing','compact','Padding','compact');
pos = 1;
for ii=prm_idx
    for jj=prm_idx
%         pos = (ii-1)*p + jj;
        if ii==jj
            ax = nexttile(pos);
            % 1D marginal -- ODE
            samps1 = zeros(n,c);
            samps1(:,:) = samps(:,ii,:);
            [f,xi] = ksdensity(samps1(:));
            prm_min(ii) = min(prm_min(ii), min(xi));
            prm_max(ii) = max(prm_max(ii), max(xi));
%             plot(xi,f)
            area(xi,f,'EdgeColor','none','FaceColor',[0 0.4470 0.7410],'FaceAlpha',0.5)
            xticks('')
            yticks('')
            grid on
            ax.Layer = 'top';
            ax.Box = 'on';
            hold on
            % 1D marginal -- eMoM
            samps_eMoM1 = zeros(n,c);
            samps_eMoM1(:,:) = samps_eMoM(:,ii,:);
            [f,xi] = ksdensity(samps_eMoM1(:));
            prm_min(ii) = min(prm_min(ii), min(xi));
            prm_max(ii) = max(prm_max(ii), max(xi));
%             plot(xi,f)
            area(xi,f,'EdgeColor','none','FaceColor',[0.8500 0.3250 0.0980],'FaceAlpha',0.5)
            xticks('')
            yticks('')
            grid on
            ax.Layer = 'top';
            ax.Box = 'on';
            hold off
        elseif ii>jj
            ax = nexttile(pos);
            hold on
            % 2D marginal -- ODE
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
            surf(x,y,z,'EdgeColor','none','FaceColor','interp','FaceAlpha',0.75)
%             xlim([min(xi(:,1)), max(xi(:,1))])
%             ylim([min(xi(:,2)), max(xi(:,2))])
            view(2)
            colormap(colorMap)
            xticks('')
            yticks('')
            grid on
            ax.Layer = 'top';
            ax.Box = 'on';
            
            % 2D marginal -- eMoM
            freezeColors
            samps_eMoM1 = zeros(n,c);
            samps_eMoM1(:,:) = samps_eMoM(:,jj,:);
            samps_eMoM2 = zeros(n,c);
            samps_eMoM2(:,:) = samps_eMoM(:,ii,:);
            [f,xi] = ksdensity([samps_eMoM1(:), samps_eMoM2(:)]);

            prm_min(jj) = min(prm_min(jj), min(xi(:,1)));
            prm_max(jj) = max(prm_max(jj), max(xi(:,1)));

            prm_min(ii) = min(prm_min(ii), min(xi(:,2)));
            prm_min(ii) = max(prm_max(ii), max(xi(:,2)));


            [x,y,z] = computeGrid(xi(:,1), xi(:,2), f);
            surf(x,y,z,'EdgeColor','none','FaceColor','interp','FaceAlpha',0.75)
            xlim([min(xi(:,1)), max(xi(:,1))])
            ylim([min(xi(:,2)), max(xi(:,2))])
            view(2)
            colormap(colorMap2)
            xticks('')
            yticks('')
            grid on
            ax.Layer = 'top';
            ax.Box = 'on';
            ax2.Visible = 'off';
            ax2.XTick = [];
            ax2.YTick = [];
            hold off
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
            xlim([prm_min(jj), prm_max(jj)]);
            L = ax.XLim;
            xspan = L(2) - L(1);
            offset = 0.15 * xspan;
            xlo = L(1) + offset;
            xhi = L(2) - offset;
            xticks([xlo xhi])
            xticklabels({'', ''})
            
        elseif ii>jj
            ax = nexttile(pos);
            xlim([prm_min(jj) prm_max(jj)]);
            ylim([prm_min(ii), prm_max(ii)]);

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

% cb = colorbar('Ticks',[0.1, 0.9],...
%     'TickLabels',{'Low probability','High probability'},...
%     'FontSize', 18);
% cb.Layout.Tile = 'south';
% cb.TickLength = 0;

title(t,'Marginal Posterior Distributions','FontSize',24)

tmp = axes;
plot(tmp,nan,nan,'color',[0 0.4470 0.7410],'DisplayName','ODE','LineWidth',20)
hold on
plot(tmp,nan,nan,'color',[0.8500 0.3250 0.0980],'DisplayName','eMoM','LineWidth',20)
legend('Location','NorthEast','FontSize',32)
tmp.Visible = 'off';

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




