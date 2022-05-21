%%
clearvars
clc
close all

%%
run("./HPO4data.m")

%%
run("./plot_MAP_prm.m")


%%
s=2:450;
s=s';
particle_diameters = 0.3000805 .* s .^ (1/3);


%% Plot TEM data vs simulation
close all
plot_tem(S2,tem_sol0,particle_diameters, s);
title("t = 1.5 hours")
plot_tem(S3, tem_sol1, particle_diameters, s);
title("t = 3.25 hours")
plot_tem(S4, tem_sol2, particle_diameters, s);
title("t = 5.0 hours")
plot_tem(S5, tem_sol3, particle_diameters, s);
title("t = 10.0 hours")



%%
function plot_tem(data, concentrations, diameters, sizes)
    figure
    hold on
    histogram(data,38,'BinLimits',[0.4,2.3],'Normalization','pdf','FaceAlpha',0.4)
    conc_transform = concentrations ./ (sizes .^ (-2/3));
    sol_avg = (conc_transform(1:2:end-1) + conc_transform(2:2:end))/2;
    diam_avg = (diameters(1:2:end-1) + diameters(2:2:end))/2;
    area = trapz(diam_avg, sol_avg);
    sol_plot = sol_avg/area;
    scatter(diam_avg, sol_plot)
    xlim([0, 2.3])
    ylim([0, 2.5])
    hold off


end
