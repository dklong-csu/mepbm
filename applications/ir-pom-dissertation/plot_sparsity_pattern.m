clearvars
clc
close all
warning('off')


figure('Position',[100 100 1300 1000])
t = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

% title(t,"Jacobian Sparsity Pattern",'FontSize',24)

%% No agglom
nexttile

r = [];
c = [];

%% A deriv
% wrt A
r(end+1) = 1;
c(end+1) = 1;

% wrt As
r(end+1) = 1;
c(end+1) = 2;

% wrt L
r(end+1) = 1;
c(end+1) = 3;

% wrt B
for i=4:2501
    r(end+1) = 1;
    c(end+1) = i;
end

%% As deriv
% wrt A
r(end+1) = 2;
c(end+1) = 1;

% wrt As
r(end+1) = 2;
c(end+1) = 2;

% wrt L
r(end+1) = 2;
c(end+1) = 3;

%% L deriv
% wrt A
r(end+1) = 3;
c(end+1) = 1;

% wrt As
r(end+1) = 3;
c(end+1) = 2;

% wrt L
r(end+1) = 3;
c(end+1) = 3;

% wrt B
for i=4:2501
    r(end+1) = 3;
    c(end+1) = i;
end

%% B3 deriv
% wrt A
r(end+1) = 4;
c(end+1) = 1;

% wrt As
r(end+1) = 4;
c(end+1) = 2;

% wrt B3
r(end+1) = 4;
c(end+1) = 4;

%% Bi deriv
% wrt A
for i=5:2501
    r(end+1) = i;
    c(end+1) = 1;
end

% wrt B
for i=5:2501
    r(end+1) = i;
    c(end+1) = i-1;

    r(end+1) = i;
    c(end+1) = i;
end

%%
v = ones(1,length(c));
A = sparse(r,c,v,2501, 2501);
spy(A)
yticks('')
xticks('')
jf = java.text.DecimalFormat;
numOut = jf.format(nnz(A));
xlabel(strcat("number non-zero = ",char(numOut)),'FontSize',18)
title('No agglomeration','FontSize',18)

%% Light agglom
nexttile

for i=4:101
    for j=4:101
        r(end+1) = i;
        c(end+1) = j;
    end
end

v = ones(1,length(c));
A = sparse(r,c,v,2501, 2501);
spy(A)
yticks('')
xticks('')
jf = java.text.DecimalFormat;
numOut = jf.format(nnz(A));
xlabel(strcat("number non-zero = ",char(numOut)),'FontSize',18)
title("Agglomeration up to size 100",'FontSize',18)


%% A lot of agglom
nexttile

for i=4:1251
    for j=4:1251
        r(end+1) = i;
        c(end+1) = j;
    end
end


v = ones(1,length(c));
A = sparse(r,c,v,2501, 2501);
spy(A)
yticks('')
xticks('')
jf = java.text.DecimalFormat;
numOut = jf.format(nnz(A));
xlabel(strcat("number non-zero = ",char(numOut)),'FontSize',18)
title("Agglomeration up to size 1250",'FontSize',18)

