function stop = plot1DBayesOptIterations(results, state)
    persistent bo_plot n_iters
    stop = false;
    % only make a graph if there is 1 optimization variable
    n_opt_vars = 0;
    vars = results.VariableDescriptions;
    for ii=1:length(vars)
        v = vars(ii);
        n_opt_vars = n_opt_vars + v.Optimize;
    end
    
    if n_opt_vars == 1
        switch state
            case 'initial'
                bo_plot = figure('Position',[100 100 1300 1000]);
            case 'iteration'
                figure(bo_plot)
                vars = results.VariableDescriptions;
                use_log_scale = false;
                for ii=1:length(vars)
                    v = vars(ii);
                    if v.Optimize
                        if strcmp(v.Transform,'log')
                            vals = 10.^(linspace(log10(v.Range(1)), log10(v.Range(2)),500));
                            use_log_scale = true;
                        else
                            vals = linspace(v.Range(1), v.Range(2),500);
                        end
                        XTable = table(vals','VariableNames',{v.Name});
                        break
                    end
                end
                [mu,sigma] = predictObjective(results,XTable);

                plot(vals,mu)
                hold on
                shade(vals,mu-2*sigma,vals,mu+2*sigma,...
                    'FillType',[1 2; 2 1],...
                    'FillColor','black',...
                    'FillAlpha',.2,...
                    'LineStyle',"none")
                scatter(results.XTrace.Variables, results.ObjectiveTrace,50,'blue','filled')
                next_x = results.NextPoint;
                next_y = predictObjective(results,next_x);
                scatter(next_x.Variables, next_y,50,'black','filled')
                if use_log_scale
                    set(gca,'XScale','log')
                end
                hold off
        end
    end
end