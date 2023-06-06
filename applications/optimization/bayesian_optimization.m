clearvars
clc
close all
uqlab -nosplash
rng('default')

% Perform Bayesian optimization
% For now, stopping criteria is number of function evaluations
min_fcn_evals = 2;
max_fcn_evals = 10;

% Begin by evaluating function at corners of domain
lb = [0.5];
ub = [2.5];
centroid = mean([lb;ub]);

lb_ub_combo = all_endpoints(lb,ub);
eval_pts = [lb_ub_combo; centroid];

fprintf("The initial evaluation points are:\n")
for iii=1:size(eval_pts,1)
    fprintf("%f ", eval_pts(iii,:))
    fprintf("\n")
end

fcn_evals = zeros(size(eval_pts,1),1);
for iii=1:length(fcn_evals)
    fcn_evals(iii) = eval_cost(eval_pts(iii,:));
end

% Enter optimization algorithm
tic
do_opt = true;
iter = 1;
while do_opt
    fprintf("Running BO iteration %d...\n",iter)
    % Train the surrogate model on the evaluation points
    GP_model = train_surrogate(eval_pts, fcn_evals);

    % Define the acquisition function
%     acquisition_fcn = @(x) confidence_bound(x, GP_model, -3);
    acquisition_fcn = @(x) -probability_of_improvement(x, GP_model, min(fcn_evals),eval_pts);
%     acquisition_fcn = @(x) -expected_improvement(x, GP_model, min(fcn_evals));;
    acq = acquisition_fcn([lb(1):0.01:ub(1)]');
    hold on
    yyaxis right
    plot(lb(1):0.01:ub(1),acq,'--r','DisplayName','Aquisition function')
    title_str = strcat("Iteration ", num2str(iter));
    title(title_str)
    xlabel('')
    ylabel('')
    yyaxis left
    hold off

    % Global optimization on the acquisition function to choose next
    % evaluation point
    fprintf("Optimizing acquisition function...")
    [new_eval_pt,~] = find_global_opt(acquisition_fcn, lb, ub);
    fprintf("New evaluation point is:\n")
    fprintf("%f ", new_eval_pt)
    fprintf("\n")

    % Evaluate the cost function at the chosen point 
    fprintf("Evaluating cost function at new point...")
    new_fcn_eval = eval_cost(new_eval_pt);

    eval_pts = [eval_pts; new_eval_pt];
    fcn_evals = [fcn_evals; new_fcn_eval];
    fprintf("done!\n")

    % Check termination condition
    [mu, sigma2] = uq_evalModel(GP_model, eval_pts);
    ucb = min(mu+5*sqrt(sigma2));
    lcb_fcn = @(x) lcb_calc(GP_model, x, 5);
    [~,lcb] = find_global_opt(lcb_fcn, lb, ub);
    regret = ucb - lcb;
    hold on
        plot(lb(1):0.01:ub(1),ucb*ones(1,length(lb(1):0.01:ub(1))),'--g','DisplayName','upper confidence bound')
        plot(lb(1):0.01:ub(1),lcb*ones(1,length(lb(1):0.01:ub(1))),'--m','DisplayName','lower confidence bound')
    hold off
    fprintf("The regret is: %f\n", regret)

    loo_var = GP_model.Internal.Error.varY;
    fprintf("LOO std is: %f\n", sqrt(loo_var))
    loo_err_dom = sqrt(loo_var) > regret;

    if iter >= max_fcn_evals
        do_opt = false;
        fprintf("Bayesian optimization terminated after %d iterations.\n",iter)
        fprintf("The maximum number of function evaluations has been reached.\n")
    elseif loo_err_dom && iter >= min_fcn_evals
        do_opt = false;
        fprintf("Bayesian optimization terminated after %d iterations.\n",iter)
        fprintf("The statistical noise in the parameters of the surrogate model dominates the variance in the evaluation of the model.\n")
    end
    iter = iter+1;
end
toc

[min_cost, idx] = min(fcn_evals);
minimizer = eval_pts(idx,:);
fprintf("The optimal parameters are:\n");
fprintf("%f ", minimizer)
fprintf("\n")


%%
figure
x_plot = lb(1):.01:ub(1);
y_plot = 0*x_plot;
for iii=1:length(x_plot)
    y_plot(iii) = eval_cost(x_plot(iii));
end
plot(x_plot,y_plot)
title('Example cost function')
% xlim([0 6])
% ylim([-6 2])

%% TEMP
function M = train_surrogate(pts, responses)
    MetaOpts.Type = 'Metamodel';
    MetaOpts.MetaType = 'Kriging';
    MetaOpts.Display = 'quiet';
    
    MetaOpts.ExpDesign.Sampling = 'User';
    MetaOpts.ExpDesign.X = pts;
    MetaOpts.ExpDesign.Y = responses;

    MetaOpts.Trend.Type = 'linear';

    MetaOpts.Corr.Family = 'Matern-5_2';

    MetaOpts.Regression.SigmaNSQ = 'auto';

    M = uq_createModel(MetaOpts);
    %uq_print(M)
    uq_display(M)
end


function C = eval_cost(x)
    C = sin(10*pi*x)/(2*x) + (x-1)^4;
end


function y = some_fcn(x, surrogate)
    [mu, sigma2] = uq_evalModel(surrogate, x);
    y = mu - 3*sqrt(sigma2);
end


function [minimizer, fval] = find_global_opt(fcn, lb, ub)
    hybridOpts = optimoptions('fmincon','OptimalityTolerance',1e-10,'Display','off');
    options = optimoptions('ga','HybridFcn',{'fmincon',hybridOpts},'UseVectorized',true,'Display','off');
    [minimizer, fval] = ga(fcn,length(lb),[],[],[],[],lb,ub,[],options);
end


function y = form_next_dim(x,lb,ub,pt_list)
    dim = length(x);
    if dim+1 == length(lb)
        y = [pt_list; x lb(dim+1); x ub(dim+1)];
    else
        y = form_next_dim([x lb(dim+1)],lb,ub,pt_list);
        y = form_next_dim([x ub(dim+1)],lb,ub,y);
    end
end


function y = all_endpoints(lb,ub)
    x = [];
    pt_list = [];
    y = form_next_dim(x,lb,ub,pt_list);
end

function y = lcb_calc(surrogate, x, beta)
    [mu, sigma2] = uq_evalModel(surrogate, x);
    y = mu - beta * sqrt(sigma2);
end