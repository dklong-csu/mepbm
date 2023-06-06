%%
close all
clc
clearvars
%%
for i=1:11
    filename = strcat('./BO_figs/lcb_1D_interp/BO_iteration_',num2str(i),'.fig');
    f = openfig(filename);
    f.Position(3:4)=[1100,800];
    legend('Location','NorthWest')
    ylim([-100 800])
    xlabel('Parameter')
    ylabel('Cost')
    frame = getframe(f);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,512);
    if i == 1
        imwrite(imind,cm,'amgen_gp.gif','gif','Loopcount',Inf,'DelayTime',1);
    elseif i == 11
        imwrite(imind,cm,'amgen_gp.gif','gif','WriteMode','append','DelayTime',5);
    else
        imwrite(imind,cm,'amgen_gp.gif','gif','WriteMode','append','DelayTime',1);
    end
    close all
end