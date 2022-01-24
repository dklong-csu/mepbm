function print_formatted_stats(vector, var)
    m = mean(vector);
    s = std(vector);
    
    if m < 1000
        fprintf("%s^\\ast=%.0f \\pm %.0f\n",var,round(m),round(s))
    else
        order = floor(log10(m));
        m_log = m / (10^order);
        s_log = s / (10^order);
        fprintf("%s^\\ast=(%.2f \\pm %.2f)\\times 10^%.0f\n",var,m_log,s_log,order)
    end
end