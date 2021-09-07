function negsemilogy(x,y)
    % Do log10 but keep sign
    ylog = sign(y).*log10(abs(y));
    % Just to get axis limits
    plot(x,ylog,'o')
    % Get limits
    lims = ylim;
    wdth = diff(lims);
    % Wrap negative data around to positive side
    ylog(ylog<0) = ylog(ylog<0) + wdth;
    % Plot
    plot(x,ylog,'o')
    % Mess with ticks
    tck = get(gca,'YTick')';
    % Shift those that were wrapped from negative to positive (above) back 
    % to their original values
    tck(tck>lims(2)) = tck(tck>lims(2)) - wdth;
    % Convert to string, then remove any midpoint
    tcklbl = num2str(tck);
    tcklbl(tck==lims(2),:) = ' ';
    % Update tick labels
    set(gca,'YTickLabel',tcklbl)
end