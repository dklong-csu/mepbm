function ac = multidim_autocovariance(samples, lag)
    n = size(samples,1);
    ac = 0;
    denom = 0;
    m = mean(samples);
    for i=1:n-lag
        x = samples(i,:)-m;
        y = samples(i+lag,:)-m;
        ac = ac + x'*y;
        denom = denom + 1;
    end
    ac = ac/denom;
end