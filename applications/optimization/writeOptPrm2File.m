function writeOptPrm2File(X, desired_size, filename)
% Check possible variables in X, otherwise use default values
  try
    A0 = X.A0;
  catch
    A0 = 0.0012; % From experiment where we have data
  end
  
  try
    tend = X.tend;
  catch
    tend = 4.838;
  end
  
  try
    drip_rate = X.drip_rate;
  catch
    drip_rate = 0;
  end

  try
    drip_time = X.drip_time;
  catch
    drip_time = 0;
  end

  try
    POM0 = X.POM0;
  catch
    POM0 = 0;
  end

  try
    solvent = X.solvent;
  catch
    solvent = 11.3;
  end

  try
    kf_mult = X.kf_mult;
  catch
    kf_mult = 1;
  end

  try
    k1_mult = X.k1_mult;
    k2_mult = X.k1_mult;
    k3_mult = X.k1_mult;
  catch
    k1_mult = 1;
    k2_mult = 1;
    k3_mult = 1;
  end

%   try
%     k2_mult = X.k2_mult;
%     k3_mult = X.k2_mult;
%   catch
%     k2_mult = 1;
%     k3_mult = 1;
%   end

%   try
%     k3_mult = X.k3_mult;
%   catch
%     k3_mult = 1;
%   end


  % Write to the input file
  fileID = fopen(filename,'w');
  vars = [A0, tend, drip_rate, drip_time, POM0, solvent, desired_size, kf_mult, k1_mult, k2_mult, k3_mult];
  fprintf(fileID,"%.20f\n",vars);
  fclose(fileID);
end