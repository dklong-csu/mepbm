function out = writeDesignPrm2File(design, filename)
% Write to the input file
fileID = fopen(filename,'w');
fprintf(fileID,"%.20f\n",design);
fclose(fileID);
end