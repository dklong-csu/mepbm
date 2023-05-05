clearvars
close all
clc

delete(gcp);
parpool(96);
rng default

%%
addpath(fullfile('..','..','lib','matlab'));
%%
tic
n_total = 1000;
log_ratio = zeros(1,n_total);
WaitMessage = parfor_wait(n_total, 'Waitbar',true);
parfor i=1:n_total
    lambda = optimizableVariable('lambda',[1e-5, 1e0],'Transform','log');
    % pltFcns = {@plotAcquisitionFunction, @plot1DBayesOptIterations};
    pltFcns = {};
    max_cost_evals = 25;
    acquisitionFcn = 'expected-improvement-plus';

    n_data = 1000;
    pretendNumer = makePretendNumerData(n_data);
    pretendDenom = makePretendDenomData(n_data);
    pretendData = [pretendNumer, pretendDenom];

    opt_fcn = @(X) 1-doKFoldCrossVal(pretendData, X.Variables, 2);

    results = bayesopt(opt_fcn, lambda, ...
      'PlotFcn', pltFcns,...
      'IsObjectiveDeterministic', true,...
      'MaxObjectiveEvaluations',max_cost_evals,...
      'Verbose',0,...
      'AcquisitionFunctionName',acquisitionFcn,...
      'OutputFcn',@BOoutputFcn);
    opt_lambda = results.XAtMinEstimatedObjective.Variables;


    opt_fcn = @(beta) calcCostFcn(beta, opt_lambda, pretendNumer, pretendDenom);
    options = optimoptions(@particleswarm,'Display','off','UseVectorized',true);
    beta = particleswarm(opt_fcn, size(pretendNumer,2),[],[],options);

    pretendTestData = makePretendNumerData(1);
    log_ratio(i) = dot(beta, pretendTestData);
    
    WaitMessage.Send;
end
toc
WaitMessage.Destroy;
%%
probs = 1 ./ ( 1 + exp(-log_ratio) );
figure('Position',[100 100 1300 1000])
boxplot(probs)

%%
function stop = BOoutputFcn(results, state)
    persistent n_iters
    stop = false;
    switch state
        case 'initial'
            n_iters = 1;
        case 'iteration'
            if n_iters > 5 && results.MinObjective < 0.1
                stop = true;
            end
            n_iters = n_iters + 1;
    end         
end


function data = makePretendNumerData(n_data)
    probs = 1:2;
    probs = probs / sum(probs);
    pd = makedist('Multinomial','probabilities',probs);
    data = zeros(n_data, length(probs));
    for ii=1:n_data
        r = random(pd, 300, 1);
        for jj=1:length(probs)
            n = length(r( r == jj));
            data(ii,jj) = n / length(r);
        end
    end
end

function data = makePretendDenomData(n_data)
    probs = ones(1,2);
    probs = probs / sum(probs);
    pd = makedist('Multinomial','probabilities',probs);
    data = zeros(n_data, length(probs));
    for ii=1:n_data
        r = random(pd, 300, 1);
        for jj=1:length(probs)
            n = length(r( r == jj));
            data(ii,jj) = n / length(r);
        end
    end
end

function J = calcJfcn(beta, data1, data2)
    M = size(data1,1);
    N = size(data2,1);
    J = zeros(size(beta,1),1);
    for ii=1:M
        J = J + log( 1 + N/M * exp(-sum(beta.*data1(ii,:),2) ) );
    end
    
    for ii =1:N
        J = J + log( 1 + M/N * exp(sum(beta.*data2(ii,:),2) ) );
    end
    J = J / (M+N);
end


function cost = calcCostFcn(beta, lambda, data1, data2)
    cost = calcJfcn(beta, data1, data2) + lambda * sum( abs(beta),2 );
end


function p = classifyData(nu,beta,data)
    d = 1 + nu * exp(-dot(beta,data));
    p = 1/d;
end

function percentCorrect = checkPredictions(beta,data1,data2)
    % data1 should be mapped to 1
    % data2 should be mapped to 0
    n_correct = 0;
    n_total = size(data1,1) + size(data2,1);
    nu = size(data2,1) / size(data1,1);
    for ii=1:size(data1,1)
        p = classifyData(nu,beta,data1(ii,:));
        n_correct = n_correct + (p >= 0.5);
    end
    
    for ii=1:size(data2,1)
        p = classifyData(nu, beta, data2(ii,:));
        n_correct = n_correct + (p < 0.5);
    end
    percentCorrect = n_correct / n_total;
end

function accuracy = crossvalFcn(Xtrain, Xtest, lambda)
    % X is formatted so first half of columns are from numerator and the
    % second half of the columns are from denominator
    n_cols = size(Xtrain,2);
    data1 = Xtrain(:,1:n_cols/2);
    data2 = Xtrain(:,n_cols/2+1:end);
    
    % optimize beta for given lambda
    n_beta = n_cols/2;
    
    opt_fcn = @(beta) calcCostFcn(beta, lambda, data1, data2);
    options = optimoptions(@particleswarm,'Display','off','UseVectorized',true,'MaxIterations',500);
    beta = particleswarm(opt_fcn, n_beta,[],[],options);
    
    % Test accuracy
    train1 = Xtest(:,1:n_cols/2);
    train2 = Xtest(:,n_cols/2+1:end);
    accuracy = checkPredictions(beta, train1, train2);
end

function avgAccuracy = doKFoldCrossVal(X, lambda, K)
    myFcn = @(Xtrain, Xtest) crossvalFcn(Xtrain, Xtest, lambda);
    options = statset('UseParallel',false);
    accuracies = crossval(myFcn, X,'KFold',K,'options',options);
    avgAccuracy = mean(accuracies);
end