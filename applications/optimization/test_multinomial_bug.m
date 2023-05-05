clearvars
%clc

p = gcp('nocreate');
if isempty(p)
    parpool(96);
elseif p.NumWorkers < 96
    delete(gcp);
    parpool(96);
end

n_time_points = 4;
n_draws = 1000;
n_data_sims_per_draw = 1000;
n_particles = 2498;

tic
partialData = cell(n_draws, n_time_points);
% diams = atoms_to_diam(3:2500);
for tt=0:n_time_points-1
    filename = strcat("design_prm_t",num2str(tt),".out");
    fileID = fopen(filename,'r');
    formatSpec = '%f';
    sizeArray = [n_particles Inf];
    PSDs = fscanf(fileID, formatSpec, sizeArray)';

    % Each row of PSDs gives the concentrations of each particle  
    slicedData = cell(n_draws,1);
    for ii=1:n_draws
        conc = PSDs(ii,:);
        conc = max(conc,0);
        probs = conc / sum(conc);
        pd = makedist('Multinomial','probabilities',probs);
        dataRowI = zeros(n_data_sims_per_draw,2);
        for jj=1:n_data_sims_per_draw % number of draws per parameter
            diams = atoms_to_diam(3:2500);
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
toc
