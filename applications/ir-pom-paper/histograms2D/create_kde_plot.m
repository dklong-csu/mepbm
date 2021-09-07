function f = create_kde_plot(data1, data2, num_pts)
    gridx1 = linspace(min(data1),max(data1),num_pts);
    gridx2 = linspace(min(data2),max(data2),num_pts);
    
    [x1, x2] = meshgrid(gridx1, gridx2);
    
    [pdf, xi] = ksdensity([data1 data2],[x1(:), x2(:)]);
    pdf_grid = reshape(pdf, num_pts, num_pts);
    f = figure;
    contour(x1, x2, pdf_grid, 10)
    colormap('copper')
    xticks('')
    yticks('')
    pbaspect([1 1 1])
    xlim([min(data1), max(data1)])
    ylim([min(data2), max(data2)])
end