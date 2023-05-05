function prob = diameters_to_bin_probability(diameters, bin_edges)
    prob = histcounts(diameters,bin_edges,'Normalization','probability');
    prob = prob/sum(prob);
end