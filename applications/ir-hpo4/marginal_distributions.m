%% Import chains
root = "/home/danny/r/mepbm/ir-hpo4/mech1_A_3step/";
samples_per_chain = 20000;
chains=0:95;
n_chains = length(chains);
samples = zeros(samples_per_chain*n_chains,6);

for i=chains
    file_name = strcat(root,"chain.",num2str(i),".txt");
    fileID = fopen(file_name, 'r');
    formatSpec = '%f %f %f %f %f %f';
    sizeChain = [6 Inf];
    chain = fscanf(fileID, formatSpec, sizeChain);
    fclose(fileID);
    chain = chain';
    samples(1+i*samples_per_chain:(i+1)*samples_per_chain,:) = chain;
end

%% Check burn-in
labels = ["k_f","k_b","k_1","k_2","k_3","M"];
for i=1:6
    figure
    hold on
   for c=chains
       chain = samples(1+c*samples_per_chain:(c+1)*samples_per_chain,:);
       plot(1:samples_per_chain,chain)
   end
   title(labels(i))
   hold off
end

%% Plot marginals
labels = ["k_f","k_b","k_1","k_2","k_3","M"];
limits = [1e-4, 1e7, 1e6, 1e6, 1e7, 200];
for i=1:5
    figure
    histogram(samples(:,i),100,'Normalization','pdf','BinLimits',[0 limits(i)])
    title(labels(i))
end

figure
histogram(samples(:,end),'BinWidth',1, 'BinLimits',[0 limits(end)])
title(labels(end))

%% Find center
for i=1:6
    fprintf("mean %s: %g\n",labels(i),mean(samples(:,i)))
end

for i=1:6
    fprintf("median %s: %g\n",labels(i),median(samples(:,i)))
end

for i=1:6
    fprintf("mode %s: %g\n",labels(i),mode(samples(:,i)))
end

