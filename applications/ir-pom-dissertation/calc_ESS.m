function [ESS, rho, iat] = calc_ESS(samples)
    rho = autocorrelation(samples);
    rho_trunc = calc_lag_trunc(rho);
    [I, K, J] = size(samples);
    
    ESS = zeros(K,1);
    iat = zeros(K,1);
    for k=1:K
        iat(k) = 1 + 2 * sum(rho_trunc{k});
        ESS(k) = J*I/iat(k);
    end
end


%% local functions
function W = calc_W(samples)
    [I, K, J] = size(samples);
    W = zeros(K,K);
    for j=1:J
        mu_j = mean(samples(:,:,j),1);
        mu_j = reshape(mu_j, K, 1);
        for i=1:I
            samp = samples(i,:,j);
            samp = reshape(samp,K,1);
            W = W + (samp - mu_j) * (samp - mu_j)';
        end
    end
    W = W / (J*(I-1));
end


function B = calc_B(samples)
    [I, K, J] = size(samples);
    mu = mean(samples,[1 3]);
    mu = reshape(mu,K,1);
    B = 0 * mu * mu';
    for j=1:J
        mu_j = mean(samples(:,:,j),1);
        mu_j = reshape(mu_j,K,1);
        B = B + (mu_j - mu) * (mu_j - mu)';
    end
    B = B * I / (J-1);
end


function V = calc_Vhat(samples)
    [I, ~, J] = size(samples);
    V = (I-1)/I * calc_W(samples) + (J+1)/(J*I) * calc_B(samples);
end


function v = variogram(samples)
    [I, K, J] = size(samples);
    v = zeros(K,I);
    for k=1:K
        for lag=0:I-1
            for j=1:J
                for i=lag+1:I
                    v(k,lag+1) = v(k,lag+1) + ( samples(i,k,j) - samples(i-lag,k,j) ).^2;
                end
            end
            v(k,lag+1) = 1/(J*(I-lag)) * v(k,lag+1);
        end
    end
end


function rho = autocorrelation(samples)
    V = calc_Vhat(samples);
    vkt = variogram(samples);
    K = size(samples,2);
    rho = vkt;
    for k=1:K
        rho(k,:) = 1 - vkt(k,:)/(2*V(k,k));
    end
end

function rho_trunc = calc_lag_trunc(rho)
    [K,I] = size(rho);
    rho_trunc = cell(K,1);
    for k=1:K
        t=1;
        test = rho(k,2*t-1+1)+rho(k,2*t-1+2);
        while test >= 0 && 2*t+1 < I
            t = t+1;
            test = rho(k,2*t-1+1)+rho(k,2*t-1+2);
        end
        rho_trunc{k} = rho(k,1:2*t-1);
    end
end