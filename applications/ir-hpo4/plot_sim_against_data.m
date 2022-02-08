%%
HPO4data

%%
precursor_simulation
tem_simulation

%% Plot precursor
figure
hold on
scatter(CH(:,1), A_conc_sim)
scatter(CH(:,1), CH(:,2))
legend("simulation","data")
xlabel("time (hours)")
ylabel("[A]")
hold off


%% Plot TEM data vs simulation
plot_tem(S2,sol_time1,particle_diameters)
title("t = 1.5 hours")
plot_tem(S3, sol_time2, particle_diameters)
title("t = 3.25 hours")
plot_tem(S4, sol_time3, particle_diameters)
title("t = 5.0 hours")
plot_tem(S5, sol_time4, particle_diameters)
title("t = 10.0 hours")


%%
function plot_tem(data, concentrations, diameters)
    figure
    hold on
    histogram(data,38,'BinLimits',[0.4,2.3],'Normalization','pdf')
    
    sol_avg = (concentrations(1:2:end-1) + concentrations(2:2:end))/2;
    diam_avg = (diameters(1:2:end-1) + diameters(2:2:end))/2;
    area = trapz(diam_avg, sol_avg);
    sol_plot = sol_avg/area;
    scatter(diam_avg, sol_plot)
    hold off
end