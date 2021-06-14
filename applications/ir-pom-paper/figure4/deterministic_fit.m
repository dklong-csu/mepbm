TEMdata

data = { S2, S3, S4, S5 };

run('./deterministic/solutions.m');

sols = { sol_t1, sol_t2, sol_t3, sol_t4 };

export_folder = './figures/deterministic_PSD_3step/';

for t=1:length(data)
    fig = figure;
    histogram(data{t},...
                      'Normalization','pdf',...
                      'BinWidth',0.1,...
                      'BinLimits',[1.4,4.1],...
                      'EdgeColor','none',...
                      'FaceColor','#0072BD')
                  
    hold on
    histogram(data{t},...
              'Normalization','pdf',...
              'BinWidth',0.1,...
              'BinLimits',[0,1.4],...
              'EdgeColor','none',...
              'FaceColor','#0072BD',...
              'FaceAlpha',0.2)
          
    sol = sols{t};
    sol_normalized = sol/trapz(atoms_to_diam(102:2500), sol(103:end));
    scatter(atoms_to_diam(102:2500),...
            sol_normalized(103:end),...
            2,[0.8500 0.3250 0.0980],...
            'MarkerEdgeAlpha',1)
        
    scatter(atoms_to_diam(3:101),...
            sol_normalized(4:102),...
            2,[0.8500 0.3250 0.0980],...
            'MarkerEdgeAlpha',0.1)

                
    xlim([0 4.1])
    ylim([0, 1.54])

    pbaspect([1 1 1])
    xlabel('Size (nm)')

    yticks('')

    set(findall(fig,'-property','FontSize'),'FontSize',18)

    hold off
    export_file = strcat(export_folder,'t',int2str(t),'.jpg');
    exportgraphics(fig, export_file,'Resolution',1200)
    close all
end