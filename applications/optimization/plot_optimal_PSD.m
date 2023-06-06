clearvars
close all
clc

%% Draw posterior samples
rng default
n_draws = 1000;

data = load('thinned_posterior_samples.mat');
stacked_samps = data.stacked_samps;
draws = draw_from_kde(stacked_samps, n_draws);
fileID = fopen('samples4cost.txt','w');
for ii=1:size(draws,1)
    d = draws(ii,:);
    for jj=1:length(d)
        fprintf(fileID, "%.20f",d(jj));
        if jj < length(d)
            fprintf(fileID,", ");
        else
            fprintf(fileID,"\n");
        end
    end
end
fclose(fileID);

%% run c++ code 

system('./calc_size_distribution shape_prm.inp');


%% Read particle concentrations in
n_particles = 2498; %2498
fileID = fopen('shape_prm_PSD.out','r');
title_text = {'Optimization Parameters:','t_{end}, A_0, POM_0'};
formatSpec = '%f';
sizeArray = [n_particles Inf];

PSDs = fscanf(fileID, formatSpec, sizeArray);

fclose(fileID);

%% Compute mean and bounds
mean_PSD = mean(PSDs,2);
upper_PSD = max(PSDs,[],2);
lower_PSD = min(PSDs,[],2);

w_mean = dot(mean_PSD, atoms_to_diam(3:2500))/sum(mean_PSD);

%% Create plot
sizes = atoms_to_diam(3:(3+n_particles-1));

desired_size = atoms_to_diam(150);

figure('Position',[100 100 1300 1000])
hold on
plot(sizes, mean_PSD,'b', 'DisplayName', 'Average PSD','LineWidth',3)
% scatter(sizes, upper_PSD,'DisplayName', 'Upper PSD bound')
% scatter(sizes, lower_PSD, 'DisplayName','Lower PSD bound')
shade(sizes,lower_PSD,sizes,upper_PSD,'FillType',[1 2;2 1],'LineStyle','none','FillColor','k','FillAlpha',.1)
h = max(upper_PSD);
v_line = linspace(0,h);
plot(desired_size*ones(size(v_line)),v_line,'r','DisplayName','Desired size','LineWidth',3)
plot(w_mean*ones(size(v_line)), v_line,'g','DisplayName','Average size','LineWidth',3)
legend
ylim([0 Inf])
% yticks('')
xlabel('Particle size (nm)')
title(title_text)

qw{1} = plot(nan,'b','LineWidth',3);
qw{2} = plot(nan,'r','LineWidth',3);
qw{3} = plot(nan,'g','LineWidth',3);
qw{4} = plot(nan,'Color',[0 0 0 .1],'LineWidth',20);
legend([qw{:}], {'Avg PSD','Desired size','Average size','PSD uncertainty'},'location','northeast')
set(findall(gcf,'-property','FontSize'),'FontSize',30)
hold off

