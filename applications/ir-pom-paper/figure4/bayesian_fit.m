TEMdata

data = { S2, S3, S4, S5 };

files = {'random_solutions_time1.m'
         'random_solutions_time2.m'
         'random_solutions_time3.m'
         'random_solutions_time4.m'};
     
root = './';

folders = {'all_3step'
           'all_4step'};

       
export_folder = {'./figures/mean_PSD_3step/'
                 './figures/mean_PSD_4step/'};
             
for f=1:length(folders)
    path = strcat(root,folders{f},'/');
    for r=1:length(files)
        run(strcat(path, files{r}));

        fig = figure;
        histogram(data{r},...
                  'Normalization','pdf',...
                  'BinWidth',0.1,...
                  'BinLimits',[1.4,4.1],...
                  'EdgeColor','none',...
                  'FaceColor','#0072BD')
        hold on
        histogram(data{r},...
                  'Normalization','pdf',...
                  'BinWidth',0.1,...
                  'BinLimits',[0,1.4],...
                  'EdgeColor','none',...
                  'FaceColor','#0072BD',...
                  'FaceAlpha',0.2)
              
        for i=1:length(solutions)-1
            sol = solutions{i};
            sol_normalized = sol/trapz(atoms_to_diam(102:2500), sol(103:end));
            scatter(atoms_to_diam(102:2500),...
                    sol_normalized(103:end),...
                    1,'k',...
                    'MarkerEdgeAlpha',.05)
            scatter(atoms_to_diam(3:101),...
                    sol_normalized(4:102),...
                    1,'k',...
                    'MarkerEdgeAlpha',.01)
        end
        sol = solutions{end};
        sol_normalized = sol/trapz(atoms_to_diam(102:2500), sol(103:end));
        scatter(atoms_to_diam(102:2500),...
                sol_normalized(103:end),...
                1,[0.8500 0.3250 0.0980],...
                'MarkerEdgeAlpha',1)
        scatter(atoms_to_diam(3:101),...
                sol_normalized(4:102),...
                1,[0.8500 0.3250 0.0980],...
                'MarkerEdgeAlpha',.1)
cd
                
        xlim([0 4.1])
        ylim([0, 1.54])
        
        pbaspect([1 1 1])
        xlabel('Size (nm)')
        
        yticks('')
        
        set(findall(fig,'-property','FontSize'),'FontSize',18)
        
        hold off
        export_file = strcat(export_folder{f},'t',int2str(r),'.jpg');
        exportgraphics(fig, export_file,'Resolution',1200)
        close all
    end
end