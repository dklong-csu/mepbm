clear all;
figure(1);clf;
h = plot(rand(10));
COL = get(h,'Color');
close all;


s  = 11.3;
kb = 0.8e4; %7.5e3; %0.8e4
kf = kb*5e-7; %0.0012; % kb*5e-7
k1 = 1.5e5; % 5.9e5; % 1.5e5
k2 = 1.65e4; % 1.9e6; % 1.65e4
k3 = 5.63e3; % 5.6e3; % 5.63e3
delx = 0.3; 
M  = 100; % 100
J  = 2500;

n10 = 0.0012;

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
ni0 = zeros(J-2,1);
x0 = [n10 ns0 p0 ni0'];

Tfinal = 4.838; %10

opts   = odeset('RelTol',1e-10,'AbsTol',1e-12,'NonNegative',1);
%% Matlab ODE solution
tic()
[t,y] = ode15s(@(t,x) ODE_fun(t,x,s,kb,kf,k1,k2,k3,M),[0,Tfinal],x0,opts);
toc()

%% Matlab eMoM solution
dx = 0.3;
xn = dx*M^(1/3);
ni0 = zeros(M-2+4,1);
x0 = [n10 ns0 p0 ni0'];
tic()
[t1,y1] = ode15s(@(t,x) ODE_fun_MoM(t,x,s,kb,kf,k1,k2,k3,M,xn),[0,Tfinal],x0,opts);
toc()

%% C++ ODE solution
fileID = fopen('prm_simple_ode.txt','w');
fprintf(fileID,'%f\n',[kf,kb,k1,k2,k3,M]);

tic()
system('./calc_concentrations_3step_simple prm_simple_ode.txt prm_simple_psd');
toc()
conc_ODE = parse_numeric_output("prm_simple_psd-4.out")';

%% Figures
% figure(1)
% h = loglog(t1,y1(:,1:4),'o',t,y(:,1:4));
% for i = 1:4;set(h(i+4),'Color',COL{i});end;
% legend('n1 PBE','ns PBE','p PBE','n3 PBE','n1 MB','ns MB','p MB','n3 MB')
% ylim([1e-10 2e-2]);
% xlim([1e-3,Tfinal]);
% xlabel('process time / h');
% ylabel('concentrations/ a.u.')
% 
% n = 12;
% m = 12;
% box on;
% legend box off;
% set(gcf,'PaperUnits','centimeter','PaperPositionMode','auto','PaperSize',[n,m],'PaperPosition',[0,0,n,m]);
% print(['Finke_vs_conc_' type '.pdf'],'-dpdf','-r500');


figure(2);clf;
xx = 3:J;
h1 = plot(dx*xx.^(1/3),y(end,4:end)./(dx*(xx+1/2).^(1/3)-dx*(xx-1/2).^(1/3)),'*');
hold all;
h4 = plot(dx*xx.^(1/3),conc_ODE./(dx*(xx+1/2).^(1/3)-dx*(xx-1/2).^(1/3)),'o');
xx = 3:M;
h2 = plot(dx*xx.^(1/3),y1(end,4:end-4)./(dx*(xx+1/2).^(1/3)-dx*(xx-1/2).^(1/3)),'square');
u = 3*k2*M^(2/3)*y1(:,end-4)/delx/k3;
xx = cumsum(8*0.3/9*k3*y1(1:end-1,1).*diff(t1),'reverse');
h3 = plot(xn+xx,u(1:end-1),'-');
set(h1,'Color',COL{1},'MarkerEdgeColor',COL{1});
set(h2,'Color',COL{2},'MarkerEdgeColor',COL{2});
set(h3,'Color',COL{2},'MarkerEdgeColor',COL{2},'LineWidth',2);
set(h4,'Color',COL{4},'MarkerEdgeColor',COL{4});
ylim([0,Inf]);
xlabel('size / nm');
ylabel('particle number density / a.u.')
legend('FW','FW C++','MB + PBE 1','MB + PBE 2')
n = 12;
m = 12;
box on;
legend box off;
set(gcf,'PaperUnits','centimeter','PaperPositionMode','auto','PaperSize',[n,m],'PaperPosition',[0,0,n,m]);
print(['Finke_vs_MoM_' type '.pdf'],'-dpdf','-r500');


figure(3);clf;
xx = 3:J;
h1 = plot(dx*xx.^(1/3),y(end,4:end),'*');
hold all;
h4 = plot(dx*xx.^(1/3),conc_ODE,'o');
xx = 3:M;
% h2 = plot(dx*xx.^(1/3),y1(end,4:end-4)./(dx*(xx+1/2).^(1/3)-dx*(xx-1/2).^(1/3)),'square');
u = 3*k2*M^(2/3)*y1(:,end-4)/delx/k3;
xx = cumsum(8*0.3/9*k3*y1(1:end-1,1).*diff(t1),'reverse');
% h3 = plot(xn+xx,u(1:end-1),'-');
set(h1,'Color',COL{1},'MarkerEdgeColor',COL{1});
% set(h2,'Color',COL{2},'MarkerEdgeColor',COL{2});
% set(h3,'Color',COL{2},'MarkerEdgeColor',COL{2},'LineWidth',2);
set(h4,'Color',COL{4},'MarkerEdgeColor',COL{4});
ylim([0,Inf]);
xlabel('size / nm');
ylabel('particle concentration/ mol/L')
% legend('FW','FW C++','MB + PBE 1','MB + PBE 2')
legend('FW', 'FW C++')
n = 12;
m = 12;
box on;
legend box off;
set(gcf,'PaperUnits','centimeter','PaperPositionMode','auto','PaperSize',[n,m],'PaperPosition',[0,0,n,m]);
print(['Finke_vs_MoM_' type '.pdf'],'-dpdf','-r500');


%% functions
function dx = ODE_fun(t,x,s,kb,kf,k1,k2,k3,M)
ri = @(i) 2.677*i.^(2/3);
k = @(i) k2 + (k3-k2).*(i>M);

n1 = x(1);
ns = x(2);
p  = x(3);
n3 = x(4);
ni = x(4:end);
N  = length(ni)+2;

dx = 0*x;
dx(1)       = -kf*n1*s^2 + kb*ns*p -   k1*n1*ns^2 - n1*sum(k(3:N).*ri(3:N).*ni');
dx(2)       =  kf*n1*s^2 - kb*ns*p - 2*k1*n1*ns^2;
dx(3)       =  kf*n1*s^2 - kb*ns*p +  k1*n1*ns^2 + n1*sum(k(3:N).*ri(3:N).*ni');
dx(4)       =  k1*n1*ns^2 - k2*n1*ri(3)*n3;
%dx(5:end)   =  k(3:N-1).*n1.*ri(3:N-1).*ni(1:end-1)' - k(4:N).*n1.*ri(4:N).*ni(2:end)';
dx(5:end-1) =  k(3:N-2).*n1.*ri(3:N-2).*ni(1:end-2)' - k(4:N-1).*n1.*ri(4:N-1).*ni(2:end-1)';
dx(end)     =  k(N-1).*n1.*ri(N-1).*ni(end-1)';

end


function dx = ODE_fun_MoM(t,x,s,kb,kf,k1,k2,k3,M,xn)
ri = @(i) 2.677*i.^(2/3);

n1 = x(1);
ns = x(2);
p  = x(3);
n3 = x(4);
ni = x(4:end-4);
N  = length(ni)+2;
m3 = x(end-3);
m2 = x(end-2);
m1 = x(end-1);
m0 = x(end-0);

delx = 0.3;

%flux = k2*n1*ri(N)*ni(end);
%q = flux;

q = 3*k2*M^(2/3)*ni(end)/delx/k3;

dx = 0*x;
dx(1)       = -kf*n1*s^2  + kb*ns*p -   k1*n1*ns^2 - n1*sum(k2.*ri(3:N).*ni') ....
    - 8*delx/9*n1*k3*xn^3*q - k3/delx^2*8/3*n1*m2;
dx(2)       =  kf*n1*s^2  - kb*ns*p - 2*k1*n1*ns^2;
dx(3)       =  kf*n1*s^2 - kb*ns*p +   k1*n1*ns^2 + n1*sum(k2.*ri(3:N).*ni') ....
    + 8*delx/9*n1*k3*xn^3*q + k3/delx^2*8/3*n1*m2;
dx(4)       =  k1*n1*ns^2 - k2*n1*ri(3)*n3;
dx(5:end-4) =  k2.*n1.*ri(3:N-1).*ni(1:end-1)' - k2.*n1.*ri(4:N).*ni(2:end)';

dx(end-3)   =  8*delx/9*n1*k3*xn^3*q + 3*8*k3*delx/9*n1*m2;
dx(end-2)   =  8*delx/9*n1*k3*xn^2*q + 2*8*k3*delx/9*n1*m1;
dx(end-1)   =  8*delx/9*n1*k3*xn^1*q +   8*k3*delx/9*n1*m0;
dx(end-0)   =  8*delx/9*n1*k3*xn^0*q;


end