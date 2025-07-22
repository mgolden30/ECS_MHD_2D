clear;

load("../dist.mat");

clf;

dist = dist / mean(abs(dist), "all");
imagesc(dist);
%imagesc(dist/n/n);
axis square;
set(gca, "ydir", "normal");
colorbar();
clim([0,1]);

%%
for i = 1:size(fs,1)
  clf;
  tiledlayout(1,2);
  vis( squeeze(fs(i,:,:,:)) );
  drawnow;

  %saveas(gcf, sprintf("frames/%03d.png", i) );
end

%%
idx = [276, 290];


dt = 1/256;
ministeps = 64;
T = dt*ministeps*(idx(2) - idx(1))

tiledlayout(2,2);  

for i = 1:2
  vis( squeeze(fs(idx(i),:,:,:)) );
end


%%

n = 256;
k = 0:n-1;
k(k>n/2) = k(k>n/2) - n;
k = reshape(k, 1, []);

%Load all the frames before animating
frames = 136;%64; %30
fs = zeros(2,n,n,frames);
for i = 1:frames
  i
  data = load("../traj/" + (i-1) + ".mat");
  fs(:,:,:,i) = data.f;
end

%Construct some 2D observables to plot
w_sq = squeeze(mean( fs(1,:,:,:).^2, [2,3] )); %mean w^2
j_sq = squeeze(mean( fs(2,:,:,:).^2, [2,3] )); %mean j^2

%append the inital values to close the loop
w_sq(end+1) = w_sq(1);
j_sq(end+1) = j_sq(1);


%%
figure(1);
while(true)
for i = 1:2:frames
  i
  %data = load("../traj/" + i + ".mat");

  f = fs(:,:,:,i);

  f = fft(f, [], 2);
  f = f .* exp( -1i* i/frames *data.sx * k );
  f = real(ifft(f, [], 2));

  clf;
  tiledlayout(1,3);
  vis(f);

  colormap bluewhitered

  nexttile
  plot( w_sq, j_sq, "Color", "black" );
  xlabel("$\langle \omega^2 \rangle$", "Interpreter", "latex" , "fontsize", 32  );
  ylabel("$\langle j^2 \rangle$", "Interpreter", "latex", "fontsize", 32 ,"rotation", 0);
  
  hold on
    scatter( w_sq(i), j_sq(i), 'o', 'filled', 'MarkerFaceColor', 'red' );
  hold off

  axis square;

  drawnow;

  %saveas(gcf, sprintf("frames/%03d.png", i) );
end
end

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
  clim([-10 10]/2);
  title("$\nabla \times {\bf u}$", "interpreter", "latex", "fontsize", fs);
  set(gca, 'ydir', 'normal'); xticks([]); yticks([]);

  nexttile
  data = squeeze(f(2,:,:)).';
  %imagesc( [data,data;data,data] );
  imagesc(data);
  axis square;
  clim([-10 10]/2);
  title("$\nabla \times {\bf B}$", "interpreter", "latex", "fontsize", fs);
  set(gca, 'ydir', 'normal'); xticks([]); yticks([]);
end