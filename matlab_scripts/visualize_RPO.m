clear;

%load("data/converged_RPO_14721.mat")

%T
%sx

%{
f = fft( f, [], 2);
f = fft( f, [], 3);
f = abs(f);
f = log10(f);
%}

make_video = true;

if make_video
  vidObj = VideoWriter('RPO_candidate.avi');
  open(vidObj);
end

enst = [];
jenst = [];

for loop = 1:4
for i = 1:4:3*128
  i
  load("traj/" + i + ".mat");
 
  
  n = size(f,2);
  f = fft(f, [], 2);
  k = 0:n-1;
  k(k>n/2) = k(k>n/2) - n;
  f = exp( sx*1j*(loop-1)*k ).*f;
  f = real(ifft( f, [], 2));

  tiledlayout(1,3);

  nexttile
  imagesc(squeeze( f(1,:,:) ).' );
  title("vorticity");
  axis square;
  colorbar();
  clim([-10 10]);
  set(gca, 'ydir', 'normal');

  nexttile
  imagesc(squeeze( f(2,:,:) ).' );
  title("current");
  axis square;
  colorbar();
  clim([-10 10]);
  set(gca, 'ydir', 'normal');
  
  addpath("derive_adjoint_looping/functions")
  colormap bluewhitered

  nexttile
  enst = [enst,  mean(f(1,:,:).^2, "all")];
  jenst= [jenst, mean(f(2,:,:).^2, "all")];

  plot(enst,jenst);
  hold on
  scatter(enst(end), jenst(end), 'filled', 'markerfacecolor', 'black');
  hold off
  axis square
  xlim([2, 10]);
  ylim([2, 10]);
  xlabel("$\langle \omega^2 \rangle$", "interpreter", "latex");
  ylabel("$\langle j^2 \rangle$", "interpreter", "latex", "rotation", 0);
  title("2D projection");


  drawnow;
  
  % Write each frame to the file.
  currFrame = getframe(gcf);
  writeVideo(vidObj,currFrame);
end
  
end
% Close the file.
close(vidObj);

