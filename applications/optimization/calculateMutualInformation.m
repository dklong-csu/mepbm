function utility = calculateMutualInformation(experimentDesign, posteriorFileName, totalSampsPerDataset)

numerFcn = @(design, n_samps) simulateData(design, posteriorFileName, totalSampsPerDataset, totalSampsPerDataset+1);
denomFcn = @(design) cell2mat(simulateData(design, posteriorFileName, totalSampsPerDataset, 1));

% Test possible experimental design variables and use default values if not
% being optimized
try
    design(1) = experimentDesign.A0;
catch
    design(1) = 0.0012;
end
try
    design(2) = experimentDesign.POM0;
catch
    design(2) = 0;
end
try
    design(3) = experimentDesign.Solvent;
catch
    design(3) = 11.3;
end
moreTimesLeft = true;
nTimes = 0;
while moreTimesLeft
    try
        design(4 + nTimes) = experimentDesign(:, strcat("Time",num2str(nTimes))).Variables;
        nTimes = nTimes + 1;
    catch
        if nTimes == 0
            design(4) = 4.838;
        end
        moreTimesLeft = false;
    end
end
            

lfireOpt.generateNumeratorData = numerFcn;
lfireOpt.generateDenominatorData = denomFcn;
lfireOpt.nMonteCarloSamples = totalSampsPerDataset;


lr = LFIRE(design, lfireOpt);
utility = mean(lr);
end