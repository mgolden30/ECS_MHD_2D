clear;

load("../debug.mat");

figure(1);
tiledlayout(4,2)

for i =1:2
nexttile
imagesc(squeeze(f0(i,:,:)).');
axis square; colorbar();
set(gca, "ydir", "normal");
title("intial condition");
end

for i =1:2
nexttile
imagesc(squeeze(f(i,:,:)).');
axis square; colorbar();
set(gca, "ydir", "normal");
title("adaptive timestep");
end


for i =1:2
nexttile
imagesc(squeeze(f2(i,:,:)).');
axis square; colorbar();
set(gca, "ydir", "normal");
title("fixed timestep");
end

diff = f - f2;

for i =1:2
nexttile
imagesc(squeeze(diff(i,:,:)).');
axis square; colorbar();
set(gca, "ydir", "normal");
end

figure(2);
semilogy(hs, 'o');
axis square;
