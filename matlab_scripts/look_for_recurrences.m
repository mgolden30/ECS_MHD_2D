%{
Let's look for PO candidates in 2D MHD
%}

clear;


load("traj.mat");

clf
imagesc(dist);
set(gca, "ydir", "normal");
colorbar();
axis square
clim([0, 0.75]);

%%

idx = [56, 69];
%idx = [97, 109];
%idx = [71, 146];
traj = reshape(traj, 2,128,128,[]);

tiledlayout(2,2)

for i = 1:2
  nexttile
  imagesc( squeeze(traj(1,:,:,idx(i))).' );
  axis square;

  nexttile
  imagesc( squeeze(traj(2,:,:,idx(i))).' );
  axis square;
end