function [thin,ESS,rho,iat] = thin_chains(samples)
    [ESS,rho,iat] = calc_ESS(samples);
    skip = ceil(max(iat));
    thin = samples(1:skip:end,:,:);
end