clearvars
close all
clc
%% Draw posterior samples
rng default
n_draws = 1000;

data = load('thinned_posterior_samples.mat');
stacked_samps = data.stacked_samps;
draws = draw_from_kde(stacked_samps, n_draws);
fileID = fopen('samples4LFIRE.txt','w');
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

[~,~] = system('./calc_size_distribution_LFIRE design_prm.inp');


%% Read particle concentrations in
n_particles = 2498; %2498
n_times = 4;
for tt=0:n_times-1
    filename = strcat("design_prm_PSD_t",num2str(tt),".out");
    fileID = fopen(filename,'r');
    title_text = strcat('Opt. Exp. Design Time ',num2str(tt));
    formatSpec = '%f';
    sizeArray = [n_particles Inf];

    PSDs = fscanf(fileID, formatSpec, sizeArray);

    fclose(fileID);

    % Compute mean and bounds
    mean_PSD = mean(PSDs,2);
    upper_PSD = max(PSDs,[],2);
    lower_PSD = min(PSDs,[],2);

    w_mean = dot(mean_PSD, atoms_to_diam(3:2500))/sum(mean_PSD);

    % Create plot
    sizes = atoms_to_diam(3:(3+n_particles-1));

    desired_size = atoms_to_diam(150);

    figure('Position',[100 100 1300 1000])
    hold on
    plot(sizes, mean_PSD,'b', 'DisplayName', 'Avg PSD','LineWidth',3)
    % scatter(sizes, upper_PSD,'DisplayName', 'Upper PSD bound')
    % scatter(sizes, lower_PSD, 'DisplayName','Lower PSD bound')
    shade(sizes,lower_PSD,sizes,upper_PSD,'FillType',[1 2;2 1],'LineStyle','none','FillColor','k','FillAlpha',.1)
    % legend
    ylim([0 Inf])
    % yticks('')
    xlabel('Particle size (nm)')
    title(title_text)
    set(findall(gcf,'-property','FontSize'),'FontSize',32)
    hold off
end