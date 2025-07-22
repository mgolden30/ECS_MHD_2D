%{
Visualize the Floquest spectrum of an RPO.
%}

clear;

load("../floquet.mat");

tiledlayout(1,2);

nexttile
imagesc(R);
axis square;
colorbar();
clim([-1,1] * max(abs(R(:))));
colorbar();
colormap bluewhitered;


nexttile
lambda = eigs(R, size(R,1));
ms = 100; %marker size
scatter( real(lambda), imag(lambda), ms, 'o', 'filled', 'MarkerFaceColor', 'red' );

theta = linspace(0,3*pi,1024);
hold on
plot( cos(theta), sin(theta), 'color', 'black', 'LineWidth', 2 );
hold off

xlim( [-1,1]*max(abs(lambda))*1.1 );
ylim(xlim);
axis square;

%% Visualize a Schur vector

tiledlayout(1,2);

k = 1;

for i = 1:2
  nexttile
  imagesc( squeeze(tang(k,i,:,:)).' );
  axis square;
  colorbar();

  clim([-1,1]* 0.02)
end