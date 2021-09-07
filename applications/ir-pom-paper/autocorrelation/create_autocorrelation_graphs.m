%% Calculate autocorrelation 3-Step

figure
xlabel('lag')
ylabel('Normalized autocovariance')

%plot_chains = randperm(232);
%plot_chains = plot_chains(1:5)-1;

hold on
for c = plot_chains
    root_folder = '../mcmc/';
    data = import_data(strcat(root_folder, 'three_step_all_times/samples/samples.'),c,1000);
    data = data(:,2:end);

    norm_factor = norm(multidim_autocovariance(data,0),'fro');
    max_lag = size(data,1)-1;
    autocov = zeros(length(0:max_lag),1);
    for i=0:max_lag
        autocov(i+1) = norm(multidim_autocovariance(data,i),'fro') / norm_factor;
    end

    plot(0:max_lag, autocov,'-')
end
hold off


%%
chains = [202,176,45,100,114];
vars = {'k_b','k_1','k_2','k_3','M'};

ar = cell(232,length(vars));

for c=0:231
    root_folder = '../mcmc/';
    data = import_data(strcat(root_folder, 'three_step_all_times/samples/samples.'),c,1000);
    data = data(:,2:end);
    for v=1:length(vars)
        [acf, lags, bounds] = autocorr(data(:,v),'NumLags',size(data,1)-1);
        ar{c+1,v} = acf;
    end
end


for v=1:length(vars)
    figure
    hold on
    for c=1:length(chains)
        plot(lags, ar{chains(c),v},'-')
    end
    
    ar_format = cat(2, ar{:,v});
    ar_avg = mean(ar_format,2);
    plot(lags, ar_avg, '-k','LineWidth',3)
    
    legend('Chain 202','Chain 176','Chain 45','Chain 100','Chain 114','Average')
    title(vars{v})
    xlabel('Lag')
    ylabel('Autocorrelation')
    hold off
    symlog('y',-1)
end

%%
chains = randperm(232);
chains = chains(1:20)-1;
vars = {'k_b','k_1','k_2','k_3','M'};

xar = cell(length(chains),1);

for c=1:length(chains)
    root_folder = '../mcmc/';
    data = import_data(strcat(root_folder, 'three_step_all_times/samples/samples.'),c,1000);
    data = data(:,2:end);
    [r, lags] = xcorr(data, size(data,1)-1,'normalized');
    xar{c} = r;
end

n = length(vars);
for i=1:n
    for j=1:n
        plot_name = strcat(vars{i},' ','vs',' ',vars{j});
        col = (i-1)*n + j;
        figure
        hold on
        for c=1:length(chains)
            plot(lags, xar{c}(:,col),'-')
        end
        title(plot_name);
        xlabel('Lag')
        ylabel(strcat('R_{',vars{i},vars{j},'}'))
        %legend('Chain 202','Chain 176','Chain 45','Chain 100','Chain 114')
        set(gca,'YScale','log')
        hold off
    end
end



