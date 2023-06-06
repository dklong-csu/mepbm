function simData = simulateData(design, posterior_samps_filename, n_draws, n_data_sims_per_draw)
% design = OED criteria
% posterior_samps_filename = where samps are stored
% n_draws = the number of posterior samples to draw
% n_data_sims_per_draw = how simulated data sets per posterior sample
writeDesignPrm2File(design, 'design_prm.txt')

data = load(posterior_samps_filename);
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

% Call the executable
[~, ~] = system("./calc_size_distribution_LFIRE");

n_particles = 2498; 

% variable number of files to read from based on number of times being
% estimated
% design = [A, POM, Solv, t1, t2, ..., N]
% so length(design) - 3 = number of times
n_time_points = length(design) - 3;
simData = cell(n_draws,1);
partialData = cell(n_draws,n_time_points);
diams = atoms_to_diam(3:2500);
for tt=0:n_time_points-1
    filename = strcat("design_prm_t",num2str(tt),".out");
    fileID = fopen(filename,'r');
    formatSpec = '%f';
    sizeArray = [n_particles Inf];
    PSDs = fscanf(fileID, formatSpec, sizeArray)';

    % Each row of PSDs gives the concentrations of each particle
    slicedData = cell(n_draws,1);
    parfor ii=1:n_draws
        conc = PSDs(ii,:);
        conc = max(conc,0);
        probs = conc / sum(conc);
        pd = makedist('Multinomial','probabilities',probs);
        dataRowI = zeros(n_data_sims_per_draw,2);
        for jj=1:n_data_sims_per_draw % number of draws per parameter
            r = random(pd, 300, 1);
            drawnDiameters = diams(r);
            dataRowI(jj, 1) = mean(drawnDiameters);
            dataRowI(jj, 2) = var(drawnDiameters);
        end
        slicedData{ii} = dataRowI;
    end
    for dd=1:n_draws
        partialData{dd,tt+1} = slicedData{dd};
    end


    fclose(fileID);
end

for ii=1:n_draws
    dataTogether = [];
    for tt=1:n_time_points
        dataTogether = [dataTogether, partialData{ii,tt}];
    end
    simData{ii} = dataTogether;
  
end