function nums = parse_numeric_output(output_file)
    fileID = fopen(output_file, 'r');
    nums = fscanf(fileID, '%f');
    fclose(fileID);
end