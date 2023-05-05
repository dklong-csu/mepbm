close all
clearvars
clc
%%
s  = 11.3;
kb = 0.8e4;
kf = kb*5e-7;
k1 = 1.5e5;
k2 = 1.65e4;
k3 = 5.63e3;
delx = 0.3;
M  = 100;
J  = 10000;
n10 = 0.0012;
Tfinal = 10;

type = 'a';
switch type
    case 'a'
        ymax = 1e-5;
    case 'b'
        n10 = 2*n10;
         k2 = 2*k2;
         k3 = 2*k3;
         ymax = 4e-6;
    case 'c'
         n10 = 4*n10;
         k2 = 4*k2;
         k3 = 4*k3;
         ymax = 2e-6;
end

ns0 = 0;
p0  = 0;
ni0 = zeros(M-2+4,1);
x0 = [n10 ns0 p0 ni0'];
xn = delx*M^(1/3);


opts   = odeset('RelTol',1e-10,'AbsTol',1e-12,'NonNegative',1);

tic()
[t1,y1] = ode23(@(t,x) ODE_fun_MoM(x,s,kb,kf,k1,k2,k3,M,xn),[0,Tfinal],x0,opts);
toc()

%% get PSD
u = 3*k2*M^(2/3)*y1(:,end-4)/delx/k3;
xx = cumsum(8*0.3/9*k3*y1(1:end-1,1).*diff(t1),'reverse');
x = [delx*(3:M).^(1/3)  xn+xx(end:-1:1)'];
q = [y1(end,4:end-4)./(delx*((3:M)+1/2).^(1/3)-delx*((3:M)-1/2).^(1/3))  u(end-1:-1:1)'];

q = max(q,0);

%% Interpolate values from MEPBM paper to be consistent
x_mepbm = delx*(3:2500).^(1/3);
interp = griddedInterpolant(x,q,'linear','linear'); % maybe play with interp/extrap methods
q_mepbm = interp(x_mepbm);
q_max = max(q_mepbm);
q_mepbm = max(q_mepbm, 1e-9*q_max);



%% plot PSD
figure(3);clf;
scatter(x,q);
ylim([0,Inf]);
hold on
scatter(x_mepbm,q_mepbm)
legend('eMoM Lukas','Match MEPBM sizes')
hold off

%% Bin conc into size range probabilities for likelihood
bin_probs = sim_concentration_to_bin_probability(q_mepbm,x_mepbm,1.4:0.1:4.1);
% Load data
TEMdata
% bin data
bin_data = histcounts(S5,1.4:0.1:4.1);
ll = dot(bin_data, log(bin_probs));


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
dx(3)       =  kf*n1*ns^2 - kb*ns*p +   k1*n1*ns^2 + n1*sum(k2.*ri(3:N).*ni') + 8*delx/9*n1*k3*xn^3*q + k3/delx^2*8/3*n1*m2;
dx(4)       =  k1*n1*ns^2 - k2*n1*ri(3)*n3;
dx(5:end-4) =  k2.*n1.*ri(3:N-1).*ni(1:end-1)' - k2.*n1.*ri(4:N).*ni(2:end)';

dx(end-3)   =  8*delx/9*n1*k3*xn^3*q + 3*8*k3*delx/9*n1*m2;
dx(end-2)   =  8*delx/9*n1*k3*xn^2*q + 2*8*k3*delx/9*n1*m1;
dx(end-1)   =  8*delx/9*n1*k3*xn^1*q +   8*k3*delx/9*n1*m0;
dx(end-0)   =  8*delx/9*n1*k3*xn^0*q;

end