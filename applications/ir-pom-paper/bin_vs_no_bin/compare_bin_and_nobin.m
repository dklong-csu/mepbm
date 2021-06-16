%% Import TEM data
TEMdata


%% Compare probabilities
% describe data
all_data = { S2, S3, S4, S5 };
ratios = cell(length(all_data),1);
bin_likelihoods = zeros(1001,4);
no_bin_likelihoods = zeros(1001,4);

for t=1:length(all_data)
    data = all_data{t};
    
    switch t
        case 1
            random_solutions_time1
        case 2
            random_solutions_time2
        case 3
            random_solutions_time3
        case 4
            random_solutions_time4
        otherwise
            error('Unexpected time point selected. Only time points 1,2,3,4 are valid.')
    end
    
    r = zeros(length(solutions)*length(data),1);
    r_index = 1;
    
    for v=1:length(solutions)
        solution = solutions{v};

        for i=1:length(data)
            % Likelihood used in paper
            % Assign particle bin via uniform bins of size 0.1 from 1.4 to 4.1
            bins = 1.4:0.1:4.1;
            diameter = data(i);
            if diameter >= 1.4 && diameter <= 4.1
                % Insert diameter into bins and sort
                bins_with_diameter = sort([bins, diameter]);
                % Find the location of the inserted particle
                % only need one instance of this and we want to find the last one in
                % case the diameter is exactly one of the bin bounds. We find the last
                % one since bins are not-inclusive on the right-endpoint (other than
                % the last bin).
                if diameter < 4.1
                    location = find(bins_with_diameter==diameter,1,'last');
                    min_diam = bins_with_diameter(location-1);
                    max_diam = bins_with_diameter(location+1);
                else
                    min_diam = bins_with_diameter(end-2);
                    max_diam = bins_with_diameter(end);
                end

                % Retrive probability of being in that bin from solution vector
                bin_index_min = max(1 + ceil(diam_to_atoms(min_diam)),4);
                bin_index_max = min(1 + floor(diam_to_atoms(max_diam)),length(solution));


                pdf = max(solution,0) / sum(max(solution(4:end),0));
                bin_prob = sum(pdf(bin_index_min:bin_index_max));

                % Likelihood proposed in appendix
                % Determine particle diameter confidence interval as size +- 0.1
                no_bin_index_min = max(1 + floor(diam_to_atoms(diameter - 0.1)),4);
                no_bin_index_max = min(1 + ceil(diam_to_atoms(diameter + 0.1)), length(solution));

                % Retrive probability of being in the diameter confidence interval
                no_bin_prob = sum(pdf(no_bin_index_min:no_bin_index_max));

                r(r_index) = bin_prob / no_bin_prob;
                r_index = r_index + 1;
                bin_likelihoods(v,t) = bin_likelihoods(v,t) + log(bin_prob);
                no_bin_likelihoods(v,t) = no_bin_likelihoods(v,t) + log(no_bin_prob);
            end
        end
    end
    ratios{t} = r;
end

%% Plot results
times = [1 2 3 4];
figure
hold on
for i=1:length(ratios)
    y = ratios{i};
    x = times(i)*ones(length(y),1);
    scatter(x,y,'bx')
end

for i=1:length(bin_likelihoods)
    scatter(5, exp(sum(bin_likelihoods(i,:)) - sum(no_bin_likelihoods(i,:))),'bx')
end
hold off
    
