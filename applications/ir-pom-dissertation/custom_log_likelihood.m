function ll = custom_log_likelihood(exec, base_file_name, params)
    [n_chains, ~] = size(params);
    for ii=1:n_chains
        inp = strcat(base_file_name,"-",num2str(ii),".inp");
        fileID = fopen(inp,'w');
        fprintf(fileID,"%.20f\n",params(ii,:));
        fclose(fileID);
    end
    system_call = strcat(exec," ",base_file_name," ",num2str(n_chains));
    [status, cmdout] = system(system_call);
    output_file = strcat(base_file_name,".out");
    ll = parse_numeric_output(output_file);
end