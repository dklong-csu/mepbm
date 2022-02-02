%% Prepare data structures
solutions_t1 = cell(3,1);
solutions_t2 = cell(3,1);
solutions_t3 = cell(3,1);
solutions_t4 = cell(3,1);
%% 
mean_sol_new
solutions_t1{1} = sol0(4:end);
solutions_t2{1} = sol1(4:end);
solutions_t3{1} = sol2(4:end);
solutions_t4{1} = sol3(4:end);

%%
mean_sol_old_partial
solutions_t1{2} = sol0(4:end);
solutions_t2{2} = sol1(4:end);
solutions_t3{2} = sol2(4:end);
solutions_t4{2} = sol3(4:end);

%%
mean_sol_old_all
solutions_t1{3} = sol0(4:end);
solutions_t2{3} = sol1(4:end);
solutions_t3{3} = sol2(4:end);
solutions_t4{3} = sol3(4:end);

%%
TEMdata

%%
plot_data_and_ode(S2, solutions_t1, 0.918)
plot_data_and_ode(S3, solutions_t2, 1.170)
plot_data_and_ode(S4, solutions_t3, 2.336)
plot_data_and_ode(S5, solutions_t4, 4.838)

%%
function diam = atoms_to_diameter(x)
    diam = 0.3000805*x.^(1/3);
end



function norm_vec = normalize_solution(vec)
    d = atoms_to_diameter(102:2500);
    conc = vec(100:end)';
    integral = trapz(d,conc);
    norm_vec = vec/integral;
end



function plot_data_and_ode(data, ode_solutions,time)
    figure
    hold on
    histogram(data, ...
              'Normalization', 'pdf',...,
              'BinWidth', 0.1,...
              'BinLimits',[1.4,4.1],...
              'EdgeColor','none',...
              'FaceColor','#0072Bd')
    for i=1:3
        s = normalize_solution(ode_solutions{i});
        scatter(atoms_to_diameter(102:2500), s(100:end)',1)
    end
    legend('Data','SUNDIALS','Published 0-95','Published all')
    title(strcat("t = ",num2str(time)," hours"))
    hold off
end
