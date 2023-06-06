function y = expected_improvement(x, surrogate, f_min)
    [mu, sigma2] = uq_evalModel(surrogate, x);
    y = (f_min - mu)*normcdf(f_min,mu,sqrt(sigma2)) ...
        + sigma2*normpdf(f_min,mu,sqrt(sigma2));
end