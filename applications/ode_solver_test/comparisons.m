%% 3-step comparisons
acc_3step
sun_3step

n = 3:2500;
d = 0.3000805*n.^(1/3);
figure
plot(d, sol_acc_3(4:end), 'o')
hold on
plot(d, sol_sun_3(4:end), 'o')
legend('accurate','sundials')
hold off

%%
acc_4step
sun_4step

n = 3:2500;
d = 0.3000805*n.^(1/3);
figure
plot(d, sol_acc_4(4:end), 'o')
hold on
plot(d, sol_sun_4(4:end), 'o')
legend('accurate','sundials')
hold off