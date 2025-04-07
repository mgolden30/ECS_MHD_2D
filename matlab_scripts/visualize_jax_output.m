clear;

addpath("derive_adjoint_looping/functions/");

load("jax_rk4.mat");
load("converged.mat");

f = permute(f, [3,2,1]);

clf
tiledlayout(1,2);

nexttile
imagesc(f(:,:,1)); 
axis square;
colorbar();
clim([-10 10])
set(gca, 'ydir', 'normal');

nexttile
imagesc(f(:,:,2)); 
axis square;
colorbar();
clim([-10 10]);
set(gca, 'ydir', 'normal');

colormap bluewhitered