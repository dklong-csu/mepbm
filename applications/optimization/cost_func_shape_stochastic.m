function cost = cost_func_shape_stochastic(X, desired_size, executable, inputfile, outputfile, posterior_samps_filename, n_draws)
    writeOptPrm2File(X, desired_size, inputfile)

    data = load(posterior_samps_filename);
    stacked_samps = data.stacked_samps;
    draws = draw_from_kde(stacked_samps, n_draws);
    fileID = fopen('samples4cost.txt','w');
    for ii=1:size(draws,1)
        d = draws(ii,:);
        for jj=1:length(d)
            fprintf(fileID, "%.20f",d(jj));
            if jj < length(d)
                fprintf(fileID,", ");
            else
                fprintf(fileID,"\n");
            end
        end
    end
    fclose(fileID);

    % Call the executable
    sys_call = strcat(executable, " ", inputfile, " ", outputfile);
    [status, cmdout] = system(sys_call);

    cost = parse_numeric_output(outputfile);
  
end