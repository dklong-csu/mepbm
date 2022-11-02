function y = confidence_bound(x, surrogate, beta)
    [mu, sigma2] = uq_evalModel(surrogate, x);
    y = mu + beta*sqrt(sigma2);
end