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
colormap jet;


%%
% Prepare the new file.
vidObj = VideoWriter('turbulence.avi');
vidObj.FrameRate = 15;
open(vidObj);


set(gcf, "color", "black");
for i = 1:size(fs,1)
  clf;
  tiledlayout(1,2);
  vis( squeeze(fs(i,:,:,:)) );
  drawnow;

  %saveas(gcf, sprintf("frames/%03d.png", i) );
  % Write each frame to the file.
  currFrame = getframe(gcf);
  writeVideo(vidObj,currFrame);
end  
% Close the file.
close(vidObj);


%%
idx = [349, 365];
idx = [195, 228];
idx = [118, 132];
idx = [58, 88];
idx = [100, 112];
idx = [189, 200];
idx = [170, 182];
idx = [101, 110];
idx = [67, 91];
idx = [90, 108];

dt = 1/256;
ministeps = 64;
T = dt*ministeps*(idx(2) - idx(1))

tiledlayout(2,2);  

for i = 1:2
  vis( squeeze(fs(idx(i),:,:,:)) );
end


%%

n = 256*2;
k = 0:n-1;
k(k>n/2) = k(k>n/2) - n;
k = reshape(k, 1, []);

%Load all the frames before animating
frames = 144;%64; %30
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

back_color = "k";
text_color = "w";

make_gif = true;
figure(1);
set(gcf, "color", back_color);


% Prepare the new file.
    vidObj = VideoWriter('peaks.avi');
    open(vidObj);

while(true)
for i = 1:4:frames
  i
  %data = load("../traj/" + i + ".mat");

  f = fs(:,:,:,i);

  f = fft(f, [], 2);
  f = f .* exp( -1i* i/frames *data.sx * k );
  f = real(ifft(f, [], 2));

  clf;
  tiledlayout(1,3);
  vis(f);

  if back_color == "w"
    colormap bluewhitered
  else
    colormap blueblackred
  end

  nexttile
  plot( w_sq, j_sq, "Color", text_color, 'linewidth', 2 );
  xlabel("$\langle \omega^2 \rangle$", "Interpreter", "latex" , "fontsize", 32  );
  ylabel("$\langle j^2 \rangle$", "Interpreter", "latex", "fontsize", 32 ,"rotation", 0);
  
  hold on
  ms = 100;
    scatter( w_sq(i), j_sq(i), ms, 'o', 'filled', 'MarkerFaceColor', 'red' );
  hold off
  set(0, 'DefaultAxesXColor', text_color)        % X axis in white
  set(0, 'DefaultAxesYColor', text_color)   
  set(gca, 'color', back_color);
  axis square;
  

  drawnow;
  if make_gif
    saveas(gcf, sprintf("frames/%03d.png", i) );
  end
 % Write each frame to the file.
       currFrame = getframe(gcf);
       writeVideo(vidObj,currFrame);
    end
  
    % Close the file.
    close(vidObj);

if make_gif
  break;
end
end



function vis(f)
  fs = 32;
  R = 1;

  nexttile
  data = squeeze(f(1,:,:)).';
  %imagesc( [data,data;data,data] );
  %data = log10(abs(fftshift(fft2(data))));
  imagesc(data);
  axis square;
  clim([-10 10]*R);
  title("$\nabla \times {\bf u}$", "interpreter", "latex", "fontsize", fs, "color", "white");
  set(gca, 'ydir', 'normal'); xticks([]); yticks([]);

  nexttile
  data = squeeze(f(2,:,:)).';
  %imagesc( [data,data;data,data] );
  %data = log10(abs(fftshift(fft2(data))));
  imagesc(data);
  axis square;
  clim([-10 10]*R);
  title("$\nabla \times {\bf B}$", "interpreter", "latex", "fontsize", fs, "color", "white");
  set(gca, 'ydir', 'normal'); xticks([]); yticks([]);

  colormap blueblackred
end