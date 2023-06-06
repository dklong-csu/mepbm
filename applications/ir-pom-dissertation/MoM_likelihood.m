function ll = MoM_likelihood(kb,kf,k1,k2,k3,M)
    %% Solve ODE at times where data are measured
    s  = 11.3;
    J = 2500;
    delx = 0.3;
    M  = floor(M);
    n10 = 0.0012;
%     Tfinal = 4.838;
    ns0 = 0;
    p0  = 0;
    ni0 = zeros(M-2+4,1);
    x0 = [n10 ns0 p0 ni0'];
    xn = delx*M^(1/3);
    dx = 0.3;


    opts   = odeset('RelTol',1e-10,'AbsTol',1e-12,'NonNegative',1);
    data_t1 = 0.918;
    data_t2 = 1.170;
    data_t3 = 2.336;
    data_t4 = 4.838;
    n_tsteps = 1000;
    t_sol = [linspace(0,data_t1,n_tsteps), linspace(data_t1 + 0.01,data_t2,n_tsteps), linspace(data_t2 + 0.01,data_t3,n_tsteps), linspace(data_t3 + 0.01,data_t4,n_tsteps)];
    [t1,y1] = ode15s(@(t,x) ODE_fun_MoM(x,s,kb,kf,k1,k2,k3,M,xn),t_sol,x0,opts);

    %% Load measured data
    TEMdata
    data = {S2, S3, S4, S5};
    ll = 0;

    %% Calculate likelihood for each data set
    for ii=1:4
        % get PSD as number density
        xn = dx*M^(1/3);
        y1_ti = y1(1:ii*n_tsteps,:);
        t1_ti = t1(1:ii*n_tsteps);
        u = 3*k2*M^(2/3)*y1_ti(1:end,end-4)/delx/k3;
        xx = cumsum(8*0.3/9*k3*y1_ti(1:end-1,1).*diff(t1_ti),'reverse');
        x = [dx*(3:M).^(1/3) xn+xx(end:-1:1)'];
        q = [y1_ti(end,4:end-4)./(dx*((3:M)+1/2).^(1/3)-dx*((3:M)-1/2).^(1/3)) u(end-1:-1:1)'];
        q = max(q,0);
        
        % convert to PSD as concentration per particle
        xx = 3:J;
        xx1 = M+1:J;
        q_fun = @(z) interp1(x,q,z,'linear',0);
        qq = [q_fun(dx*xx1.^(1/3))];
        xn = dx*xx.^(1/3);
        qn = [y1_ti(end, 4:end-4) qq.*(dx*(xx1+1/2).^(1/3)-dx*(xx1-1/2).^(1/3))];
        qn_max = max(qn);
        qn = max(qn, 1e-9*qn_max);
%         PSD{ii} = qn;
        % compare to data to get log likelihood
        bin_probs = sim_concentration_to_bin_probability(qn,xn,1.4:0.1:4.1);
        % bin data
        bin_data = histcounts(data{ii},1.4:0.1:4.1);
        ll = ll + dot(bin_data, log(bin_probs));
    end
end

%%
function dx = ODE_fun_MoM(x,s,kb,kf,k1,k2,k3,M,xn)
ri = @(i) 2.677*i.^(2/3);

n1 = x(1);
ns = x(2);
p  = x(3);
n3 = x(4);
ni = x(4:end-4);
N  = length(ni)+2;
m2 = x(end-2);
m1 = x(end-1);
m0 = x(end-0);

delx = 0.3;

q = 3*k2*M^(2/3)*ni(end)/delx/k3;

dx = 0*x;
dx(1)       = -kf*n1*s^2  + kb*ns*p -   k1*n1*ns^2 - n1*sum(k2.*ri(3:N).*ni') - 8*delx/9*n1*k3*xn^3*q - k3/delx^2*8/3*n1*m2;
dx(2)       =  kf*n1*s^2  - kb*ns*p - 2*k1*n1*ns^2;
dx(3)       =  kf*n1*s^2 - kb*ns*p +   k1*n1*ns^2 + n1*sum(k2.*ri(3:N).*ni') + 8*delx/9*n1*k3*xn^3*q + k3/delx^2*8/3*n1*m2;
dx(4)       =  k1*n1*ns^2 - k2*n1*ri(3)*n3;
dx(5:end-4) =  k2.*n1.*ri(3:N-1).*ni(1:end-1)' - k2.*n1.*ri(4:N).*ni(2:end)';

dx(end-3)   =  8*delx/9*n1*k3*xn^3*q + 3*8*k3*delx/9*n1*m2;
dx(end-2)   =  8*delx/9*n1*k3*xn^2*q + 2*8*k3*delx/9*n1*m1;
dx(end-1)   =  8*delx/9*n1*k3*xn^1*q +   8*k3*delx/9*n1*m0;
dx(end-0)   =  8*delx/9*n1*k3*xn^0*q;

end
