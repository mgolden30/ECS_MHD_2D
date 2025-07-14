%{
Visualize the trajectory of 2D MHD
%}

clear;

load("../dist2.mat");

clf;

figure(1);
dist = dist / mean(abs(dist), "all");
imagesc(dist);
%imagesc(dist/n/n);
axis square;
set(gca, "ydir", "normal");
colorbar();
clim([0,1]);

%%
figure(2);
for i = 1:size(fs,1)
  i
    
  clf;
  tiledlayout(1,2);
  vis( squeeze( fs(i,:,:,:)) );
  drawnow;

  saveas(gcf, sprintf("frames/%03d.png", i) );

  colormap bluewhitered

end
return;


%% Visualize RPO

for j  =1:100
num_frames = 18;
for i = 1:num_frames
  clf;
  tiledlayout(1,2);
  load("../traj/" + (i-1) + ".mat");
  vis( squeeze( f(:,:,:)) );
  colormap bluewhitered

  drawnow;
end
end

%% Compare before and after
clf;
tiledlayout(2,2);

load("../traj/0.mat");
vis(f);

load("../traj/" + (num_frames-1) + ".mat");
vis(f);
drawnow;

%%
clf;
i = 233;
vis( squeeze( fs(i,:,:,:)) );



%%
j_sq = mean( fs( :,2,:,:).^2, [3,4] );

plot( j_sq );


return

%%
idx = [133, 156];
idx = [181, 205];

dt = 0.005;
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
frames = 24*2
for i = 0:frames-1
  i
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