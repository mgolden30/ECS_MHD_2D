%{
Visualize the Floquest spectrum of an RPO.
%}

clear;
set(gcf,"color","w");
set(0, 'DefaultAxesXColor', 'k')        % X axis in white
set(0, 'DefaultAxesYColor', 'k')   

load("../floquet.mat");
load("../solutions/Re50/floquet/1.mat");


tiledlayout(2,2);

nexttile
imagesc(R);
axis square;
colorbar();
clim([-1,1] * max(abs(R(:))));
colorbar();
colormap bluewhitered;

nexttile

%First, draw a unit circle
theta = linspace(0,3*pi,1024);
plot( cos(theta), sin(theta), 'color', 'black', 'LineWidth', 2 );

lambda = eigs(R, size(R,1));
ms = 30; %marker size
hold on
scatter( real(lambda), imag(lambda), ms, 'o', 'filled', 'MarkerFaceColor', 'red' );
hold off

xlim( [-1,1]*max(abs(lambda))*1.1 );
ylim(xlim);
axis square;
title("Floquet multipliers \lambda");
xlabel("real(\lambda)");
ylabel("imag(\lambda)")

nexttile
imagesc(squeeze(diff(1,:,:)).');
axis square;
colorbar();
clim([-1,1] * max(abs(diff), [], "all"));
title("RPO shooting error");

nexttile
tang = reshape(tang, size(tang,1), []);
proj = tang * diff(:)/norm(diff(:));
proj = abs(proj);
plot(proj);
title("Error projection onto Schur vectors")
return;

%% Visualize a Schur vector

tiledlayout(1,2);

tang = reshape(tang, [32,2,256,256]);

k = 3;

for i = 1:2
  nexttile
  imagesc( squeeze(tang(k,i,:,:)).' );
  axis square;
  colorbar();

  clim([-1,1]* 0.02)
end

