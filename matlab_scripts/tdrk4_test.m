clear;

load("../tdrk4.mat");

names = {"$\nabla \times {\bf u}\quad$", "$\nabla \times {\bf B}\quad$"}

tiledlayout(2,2);
for i= 1:2
  nexttile;
  imagesc(squeeze(f_tdrk4(i,:,:)).');
  axis square;
  clim([-10 10]);
  set(gca, 'ydir', 'normal');
  title(names{i} + "     TDRK4", "Interpreter", "latex")
  xticks([]);
  yticks([]);
end

%If you want to look at the difference
f_rk4 = f_rk4 - f_tdrk4;

for i= 1:2
  nexttile;
  imagesc(squeeze(f_rk4(i,:,:)).');
  axis square;
  %clim([-10 10]);
  colorbar();
  set(gca, 'ydir', 'normal');
  title(names{i}+ "     RK4", "Interpreter", "latex")
  xticks([]);
  yticks([]);
end

set(gcf, "color", "w");