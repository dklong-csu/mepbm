function ratios = LFIRE(design, options)
% LFIRE Likelihood-Free Inference by Ratio Estimation
%
% FIXME
%
% Options includes
% generateNumeratorData --
% generateDenominatorData --
% nMonteCarloSamples --
% maxEvalsLambdaOpt --
% lambdaBounds --
% maybe add more later?

% Check the options structure. Use default values if not provided (and
% default available)
try
    generateNumeratorData = options.generateNumeratorData;
catch
    msg = 'Error. \nYou must provide a function that generates data for the numerator.';
    error(msg)
end
try
    generateDenominatorData = options.generateDenominatorData;
catch
    msg = 'Error. \nYou must provide a function that generates data for the denominator.';
    error(msg)
end
try
    nMonteCarloSamples = options.nMonteCarloSamples;
catch
    nMonteCarloSamples = 100;
end
try
    maxEvalsLambdaOpt = options.maxEvalsLambdaOpt;
catch
    maxEvalsLambdaOpt = 25;
end
try
    lambdaBounds = options.lambdaBounds;
catch
    lambdaBounds = [1e-5, 1];
end

% Generate the denominator data. This needs to be done only once.
% Remove nDenominatorDraws?
dataDenom = generateDenominatorData(design);

dataNumer = generateNumeratorData(design, nMonteCarloSamples);


log_ratio = zeros(1,nMonteCarloSamples);
parfor i=1:nMonteCarloSamples
    rng(i);
    lambda = optimizableVariable('lambda',lambdaBounds,'Transform','log');
    % pltFcns = {@plotAcquisitionFunction, @plot1DBayesOptIterations};
    pltFcns = {};
    acquisitionFcn = 'expected-improvement-plus';

    dataNumerIteration = dataNumer{i};
    dataNumerTrain = dataNumerIteration(1:end-1,:);
    dataNumerTest = dataNumerIteration(end,:);
    combinedData = [dataNumerTrain, dataDenom];

    opt_fcn = @(X) 1-doKFoldCrossVal(combinedData, X.Variables, 2);

    results = bayesopt(opt_fcn, lambda, ...
      'PlotFcn', pltFcns,...
      'IsObjectiveDeterministic', true,...
      'MaxObjectiveEvaluations',maxEvalsLambdaOpt,...
      'Verbose',0,...
      'AcquisitionFunctionName',acquisitionFcn,...
      'OutputFcn',@BOoutputFcn);
    opt_lambda = results.XAtMinEstimatedObjective.Variables;


    opt_fcn = @(beta) calcCostFcn(beta, opt_lambda, dataNumerTrain, dataDenom);
%     options = optimoptions(@particleswarm,'Display','off','UseVectorized',true);
%     beta = particleswarm(opt_fcn, size(dataNumerTrain,2),[],[],options);
%     options = optimoptions(@fmincon,'Display','Off','Algorithm','active-set');
%     beta = fmincon(opt_fcn,ones(1, size(dataNumerTrain,2)),[],[],[],[],[],[],[],options);
    options = optimoptions(@fminunc,'Display','Off','Algorithm','quasi-newton','StepTolerance',1e-6);
%     beta = fminunc(opt_fcn,ones(1, n_beta),[],[],[],[],[],[],[],options);
beta = fminunc(opt_fcn,ones(1, size(dataNumerTrain,2)),options);
    msg = strcat("Beta = [",num2str(beta),"]\n");
%     fprintf(msg)

    log_ratio(i) = dot(beta, dataNumerTest);
end
% utility = mean(log_ratio);
ratios = log_ratio;
end





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
%      options = optimoptions(@particleswarm,'Display','off','UseVectorized',true,'MaxIterations',500);
%      beta = particleswarm(opt_fcn, n_beta,[],[],options);
    options = optimoptions(@fminunc,'Display','Off','Algorithm','quasi-newton','StepTolerance',1e-6);
%     beta = fminunc(opt_fcn,ones(1, n_beta),[],[],[],[],[],[],[],options);
beta = fminunc(opt_fcn,ones(1, n_beta),options);
    
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