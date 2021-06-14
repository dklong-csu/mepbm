% A function that imports comma delimited .txt files that share a similar
% naming convention. This function assumes a naming convention of
% /file_prefix$some_number.txt. For example,
%       file_prefix = /my/favorite/dir/my_file
%       file_numbering_labels = [1 2 4 5 200]
% is appropriate input for a situation with a file structure like
%       my
%        |
%        +-- favorite
%            |
%            +-- dir
%                |
%                /-- my_file1.txt
%                /-- my_file2.txt
%                /-- my_file4.txt
%                /-- my_file5.txt
%                /-- my_file200.txt
%
% Input arguments:
% file_prefix --> a string containg the prefix common to each file, which
%                 includes the location of the directory and the beginning
%                 of the file name.
% file_numbering_labels --> a vector containing every numeric label that
%                           needs to be appended to the end of the file
%                           prefix in order to uniquely identify a file.
function data = import_data(file_prefix, file_numbering_labels)
    data = [];
    for i=file_numbering_labels
        data = [data; readmatrix(strcat(file_prefix,num2str(i),'.txt'))];
    end
end