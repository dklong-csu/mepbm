function draws = draw_N_from_bins(prob, n_draws)
    draws = n_draws * round(prob,2);
    [~, most_probable_idx] = sort(prob,'descend');
    [~, least_probable_idx] = sort(prob,'ascend');
    most_prob_add = 1;
    least_prob_add = 1;
    while sum(draws) < n_draws
        fprintf("Add to bin %d\n",most_probable_idx(most_prob_add))
        draws(most_probable_idx(most_prob_add)) = draws(most_probable_idx(most_prob_add)) + 1;
        most_prob_add = most_prob_add + 1;
        if most_prob_add > length(most_probable_idx)
            most_prob_add = 1;
        end
    end
         
    while sum(draws) > n_draws
        fprintf("Subtract from bin %d\n",least_probable_idx(least_prob_add))
        draws(least_probable_idx(least_prob_add)) = max(0,draws(least_probable_idx(least_prob_add)) - 1);
        least_prob_add = least_prob_add + 1;
        if least_prob_add > length(least_probable_idx)
            least_prob_add = 1;
        end
    end
end