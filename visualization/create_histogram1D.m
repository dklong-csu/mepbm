% A function that creates a one-dimensional histogram to visualize a
% probablity distribution approximated via generated samples.
%
% Input arguments:
% x_vals --> a vector containing every sample available, for example the
%            values of a certain parameter for each sample.
% labels --> a cell array containing strings for labelling the x-axis, 
%            y-axis, and title. The vector should be of the form:
%            {'x label', 'y label', 'title'}
%            and one can make these labels empty by simply providing an
%            empty string '', e.g. {'', 'y label', 'title'} will result in
%            a plot where the x-axis in unlabeled by the y-axis and title
%            are labeled.
% num_bins --> a scalar representing the desired number of histogram bars
% font_size --> a scalar representing the desired font size for the various
%               labels on the histogram plot.
% limits --> a vector containing four values which decide what values the
%            axes should cover. The order is important and should be
%            [ x min value, x max value, y min value, y max value ]
%            If one wants Matlab to figure out axes ranges automatically
%            then one can use 'Inf' or '-Inf' to indicate that. For
%            example, [ 0, 1, 0, Inf ] will make the plotted x-axis range
%            from 0 to 1 and the y-axis from 0 to a maximum value that
%            Matlab calculates automatically. Changing this to 
%            [ 0, 1, -Inf, 1 ] will make the maximum y-value displayed by 1
%            but the minimum y-value will be automatically determined.
%            Using input [ -Inf, Inf, -Inf, Inf ] will result in Matlab
%            automatically calulating all of the bounds of the plotting
%            area.
% ticks --> a cell array with two entries. The first entry specifies the
%           ticks to be used in numerically labeling the x-axis and the
%           second entry specifies how to label the y-axis. These entries
%           must either be a vector of increasing numeric values or the
%           string 'auto' which tells Matlab to determine the axis tick
%           values automatically. Alternatively, if no ticks are desired,
%           one may use an empty array [] to achieve this.
%
%
function create_histogram1D(x_vals, labels, num_bins, font_size, limits, ticks)
    % Create a histogram using the provided data
    %
    % Normalization = pdf --> ensures a consistent area under the
    % histogram so that when many histograms are made, the are comparable
    % 
    % EdgeAlpha = 0 --> This modifies the transparency of the edges around
    % each histogram bar. Often times a 'bad' model will have a much larger
    % range on the x-axis whereas the 'good' model will have  a smaller
    % range. In these cases, since the edges are by default a fixed size
    % the resulting graph will have each bar dominated by the edge rather
    % than the inside coloring, leading to inconsistent histograms. A value
    % of 0 makes the edges completely transparent, whereas a value of 1
    % makes the edge not transparent at all (and transparency vaires
    % linearly between 0 and 1). Thus, this setting can be modified for
    % problems where the values on the x-axis do not vary too much between
    % different models.
    histogram(x_vals, num_bins, 'Normalization','pdf','EdgeAlpha',0);
    
    % Label the axes and add a title
    xlabel(labels{1});
    ylabel(labels{2});
    title(labels{3}, 'FontSize', font_size);
    
    % Set the limits for the plotting area
    xlim(limits(1:2))
    ylim(limits(3:4))
    
    % Set the tick marks for the plot
    xticks(ticks{1})
    yticks(ticks{2})
    
    % Set the font size of the axes
    ax = gca;
    ax.XAxis.FontSize = font_size;
    ax.YAxis.FontSize = font_size;
    
    % Force the plotting area to be square
    pbaspect([1 1 1])
end