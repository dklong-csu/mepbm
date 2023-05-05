function cost = cost_func_shape(X, desired_size, executable, inputfile, outputfile)
  writeOptPrm2File(X, desired_size, inputfile)
  
  % Call the executable
  sys_call = strcat(executable, " ", inputfile, " ", outputfile);
  [status, cmdout] = system(sys_call);

  cost = parse_numeric_output(outputfile);
  
end