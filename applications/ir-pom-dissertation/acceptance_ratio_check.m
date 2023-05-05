function [passes_ar_check, q] = acceptance_ratio_check(BI, ar_range, quant_prob)
    tail = (1 - quant_prob)/2;
    q = quantile(BI.Results.Acceptance,[tail .5 1-tail]);
%     fprintf("For reference, AR per chain: ")
%     fprintf("%f    ",BI.Results.Acceptance)
%     fprintf("\n")
%     fprintf("For reference, quantiles: ")
%     fprintf("%f    ",q)
%     fprintf("\n")
    if q(1) < ar_range(1) || q(end) > ar_range(2)
        passes_ar_check = false;
    else
        passes_ar_check = true;
    end
end