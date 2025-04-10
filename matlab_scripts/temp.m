clear;

load("../dist.mat");



%%
clf;
n = 128;

dist = dist / mean(abs(dist), "all");
imagesc(dist);
%imagesc(dist/n/n);
axis square;
set(gca, "ydir", "normal");
colorbar();
clim([0,1]);

%%
for i = 1:128%size(fs,1)
  clf;
  tiledlayout(1,2);
  vis( squeeze(fs(i,:,:,:)) );
  drawnow;

  saveas(gcf, sprintf("frames/%03d.png", i) );
end

%%
%idx = [141, 170];
%idx = [241, 264];
idx = [243, 362];
idx = [555, 582];
idx = [1015, 1207];
idx = [1263, 1323];
idx = [243, 309];
idx= [556, 593];

idx = [155, 194];

idx = [147, 189];

%idx = [390, 450];
idx = [696, 766];

idx = [218, 268];

dt = 0.01;
ministeps = 32;
T = dt*ministeps*(idx(2) - idx(1))

tiledlayout(2,2);

for i = 1:2
  vis( squeeze(fs(idx(i),:,:,:)) );
end


%%

n = 128;
k = 0:n-1;
k(k>n/2) = k(k>n/2) - n;
k = reshape(k, 1, []);


figure(1);
%while(true)
frames = 50
for i = 0:frames-1
  load("../timeseries/" + i + ".mat");

  f = fft(f, [], 2);
  f = f .* exp( -1i* i/frames *sx * k );
  f = real(ifft(f, [], 2));

  tiledlayout(1,2);
  vis(f);

  %colormap bluewhitered

  drawnow;

  saveas(gcf, sprintf("frames/%03d.png", i) );
end
%end

%%
figure(2);
clf;
%tiledlayout(2,2);
load("timeseries/" + 0 + ".mat");
vis(f);

load("timeseries/" + 24 + ".mat");
vis(f);




function vis(f)
  fs = 32;

  nexttile
  data = squeeze(f(1,:,:)).';
  %imagesc( [data,data;data,data] );
  imagesc(data);
  axis square;
  clim([-10 10]);
  title("$\nabla \times {\bf u}$", "interpreter", "latex", "fontsize", fs);
  set(gca, 'ydir', 'normal'); xticks([]); yticks([]);

  nexttile
  data = squeeze(f(2,:,:)).';
  %imagesc( [data,data;data,data] );
  imagesc(data);
  axis square;
  clim([-10 10]);
  title("$\nabla \times {\bf B}$", "interpreter", "latex", "fontsize", fs);
  set(gca, 'ydir', 'normal'); xticks([]); yticks([]);
end