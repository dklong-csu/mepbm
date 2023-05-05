function draws = draw_from_kde(input_samps, n_draws)
    % Calculate bandwidth for kernel density estimation by Silverman's rule of thumb
    std_dev = std(input_samps);
    bw = zeros(length(std_dev));
    for ii=1:size(bw,1)
        d = size(bw,1);
        n = size(input_samps,1);
        bw(ii,ii) = ( 4 / (d+2) )^(1/ (d+4) ) *...
            n^(-1/(d+4)) *...
            std_dev(ii);
        bw(ii,ii) = bw(ii,ii)^2;
    end

    % Sample from the posterior
    % randomly select a row from the samples matrix
    rows = randi(size(input_samps,1),n_draws,1);
    % random perturbations based on bandwidth
    pert = mvnrnd(zeros(1,size(input_samps,2)),bw,n_draws);

    draws = 0*pert;
    for ii=1:n_draws
        draws(ii,:) = input_samps(rows(ii),:) + pert(ii);
        draws(ii,1:end-1) = max(0, draws(ii,1:end-1));
        draws(ii,end) = max(3, draws(ii,end));
    end

end