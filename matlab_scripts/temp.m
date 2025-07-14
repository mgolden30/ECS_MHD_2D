clear;

load("../traj.mat");



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

  %saveas(gcf, sprintf("frames/%03d.png", i) );
end
return;

%%
idx = [133, 156];
idx = [181, 205];
idx = [456, 474];

idx = [385, 416];
idx = [362, 468];

idx = [376, 412]

tiledlayout(2,2);

for i = 1:2
  vis( squeeze(fs(idx(i),:,:,:)) );
end


%%

n = 128;
k = 0:n-1;
k(k>n/2) = k(k>n/2) - n;
k = reshape(k, 1, []);

% Prepare the new file.
    vidObj = VideoWriter('peaks.avi');
    open(vidObj);
    
figure(1);
%while(true)
frames = 24*2
for i = 0:frames-1
  i
  load( sprintf("../timeseries/%03d.mat", i) );

  f = fft(f, [], 2);
  f = f .* exp( -1i* i/frames *sx * k );
  f = real(ifft(f, [], 2));

  tiledlayout(1,2);
  vis(f);

  colormap bluewhitered
  set(gca, 'Color', 'w');

  drawnow;

  saveas(gcf, sprintf("frames/%03d.png", i) );
% Write each frame to the file.
       currFrame = getframe(gcf);
       writeVideo(vidObj,currFrame);
    end
  
    % Close the file.
    close(vidObj);

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
  scale = 5;


  nexttile
  data = squeeze(f(1,:,:)).';
  %imagesc( [data,data;data,data] );
  imagesc(data);
  axis square;
  clim([-1 1]*scale);
  title("$\nabla \times {\bf u}$", "interpreter", "latex", "fontsize", fs);
  set(gca, 'ydir', 'normal'); xticks([]); yticks([]);

  nexttile
  data = squeeze(f(2,:,:)).';
  %imagesc( [data,data;data,data] );
  imagesc(data);
  axis square;
  clim([-1 1]*scale);
  title("$\nabla \times {\bf B}$", "interpreter", "latex", "fontsize", fs);
  set(gca, 'ydir', 'normal'); xticks([]); yticks([]);
end