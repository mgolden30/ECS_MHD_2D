clear;

load("proj (1).mat")

ms = 100; %marker size
plt = @(x) scatter(x(1,:), x(2,:), ms, 'o', 'filled');


plt(projRPO1);
hold on
plt(projRPO2);
plt(projRPO3);
plt(projRPO4);
plt(projRPO5);
plt(projRPO6);

plt(projTW1);
plt(projTW2);
hold off
legend
