function ll = custom_log_likelihood_eMoM(params)
    [n_chains, ~] = size(params);
    ll = zeros(n_chains,1);
    parfor ii=1:n_chains
        p = params(ii,:);
        ll(ii) = MoM_likelihood(p(1),p(2),p(3),p(4),p(5),p(6));
    end
end