function y = probability_of_improvement(x, surrogate, f_min, eval_pts)
    % check if x is in list of evaluated points
    x_evald = zeros(size(x,1),1);
    for iii=1:length(x_evald)
        x_evald(iii) = sum(prod(eval_pts == x(iii,:),2));
    end
    [mu, sigma2] = uq_evalModel(surrogate, x);
    poi = normcdf(f_min, mu, sqrt(sigma2));
    
    y = (1-x_evald) .* poi;
end
