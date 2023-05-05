function prob = kde_to_bin_probability(data, bin_edges)
    [~,~,bw] = ksdensity(data);
    n_draws = 100000;
    idx = randi([1 length(data)],1, n_draws);
    jitter = normrnd(0, bw, 1, n_draws);
    draws = data(idx) + jitter;
    prob = diameters_to_bin_probability(draws, bin_edges);
end