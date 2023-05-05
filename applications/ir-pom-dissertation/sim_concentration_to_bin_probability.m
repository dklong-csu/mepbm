function prob = sim_concentration_to_bin_probability(conc,diams,bin_edges)
    prob = zeros(1, length(bin_edges)-1);
    bin = 1;
    for ii=1:length(diams)
        not_binned = true;
        d = diams(ii);

        % If the particle is too small, then we aren't binning it so move on
        if d < bin_edges(1)
            not_binned = false;
        elseif d >= bin_edges(end)
            not_binned = false;
        end

        % We don't need to reset the bin tested back to 1 because the diameters
        % are ordered such that a subsequent diameter nevers falls into a
        % previous bin
        while not_binned
            if d >= bin_edges(bin) && d < bin_edges(bin+1)
                prob(bin) = prob(bin) + conc(ii);
                not_binned = false;
            else
                bin = bin + 1;
            end
        end
    end
    prob = prob/sum(prob);
end