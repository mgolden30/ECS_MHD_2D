%{
I did some benchmarking
%}

clear;

i=1;


names{i} = "L40S";
transient(i) = 6.1969;
turb(i) = 11.918;
recurrence(i) = 1.44;
adjoint_descent(i) = 0.96;
f(i) = 0.181;
J(i) = 0.316;
Jt(i) = 0.773;
gmres(i) = 20.4;
i = i+1;


names{i} = "Quadro RTX 6000";
transient(i) = 10.7876;
turb(i) = 18.834;
recurrence(i) = 2.139;
adjoint_descent(i) = 1.49;
f(i) = 0.274;
J(i) = 0.511;
Jt(i) = 1.224;
gmres(i) = 31.8;
i = i+1;



names{i} = "H100 80GB";
transient(i) = 3.13;
turb(i) = 5.67;
recurrence(i) = 0.669;
adjoint_descent(i) = 0.75;
f(i) = 0.114;
J(i) = 0.169;
Jt(i) = 0.677;
gmres(i) = 16.1;
i = i+1;

names{i} = "V100 16GB";
transient(i) = 5.88;
turb(i) = 10.857;
recurrence(i) = 1.44;
adjoint_descent(i) = 1.119;
f(i) = 0.176;
J(i) = 0.313;
Jt(i) = 0.948;
gmres(i) = 24;
i = i+1;


names{i} = "RTX 3060";
transient(i) = 19.7925;
turb(i) = 39.24855;
recurrence(i) = 3.55;
adjoint_descent(i) = 2.8;
f(i) = 0.517;
J(i) = 1.01;
Jt(i) = 2.243;
gmres(i) = 56.4;
i = i+1;


divide = @(v) v(end)./v;

A = [ divide(transient); divide(turb); divide(recurrence); divide(adjoint_descent); divide(f); divide(J); divide(Jt); divide(gmres) ];

shape = {"o", "d", "^", "<", "s"};

clf
ms = 100;
for i = 1:size(A,2)
    i
  scatter(1:size(A,1), A(:,i), ms, 'filled', shape{i});
  hold on
end

fs = 24;

xlim([0.5, 8.5]);
ylim([0.5, 9]);

xlabel("Task", "FontSize", fs);
ylabel("Speedup", "FontSize", fs);

l = legend(names, "color", "black");
l.TextColor = "w";

set(gca, "color", "black");
set(gcf, "color", "black");

box on


ax = gca;                % get current axes
ax.XColor = 'w';         % make x-axis line white
ax.YColor = 'w';         % make y-axis line white
ax.ZColor = 'w';         % if 3D, make z-axis white
ax.Color  = 'k';         % background color (optional, e.g. black)
ax.GridColor = 'w';      % grid lines white (optional)
ax.MinorGridColor = 'w'; % minor grid lines white (optional)
ax.TickDir = 'out';      % optional styling

