% Adapted from code at: https://github.com/drhandwerk/pbm
function f = plot_tem_histogram(histdata,diameter_range,time,font_size)
    f=figure;
    color_code = '#0072BD';
    histogram(histdata,27, 'BinLimits',diameter_range,'FaceColor',color_code);
    hold on
    histogram(histdata,'BinWidth',0.1, 'BinLimits',[0 diameter_range(1)],'FaceAlpha',0.2,'FaceColor',color_code);
    xlim([0 diameter_range(2)])
    ylim([0 30])

    %pbaspect([1 1 1])
    title([num2str(time) ' hours']);
    xlabel('Size (nm)')
    ylabel('TEM data counts')
    %yticks([])
    hold off
    axis square
    
    set(findall(f,'-property','FontSize'),'FontSize',font_size)
end