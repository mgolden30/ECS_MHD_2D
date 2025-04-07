%{
Visualize a multishooting RPO
%}

clear;

make_video = true;

if make_video
  vidObj = VideoWriter('RPO_candidate.avi');
  open(vidObj);
end

enst = [];
jenst = [];

%for loop = 1:8
loop = 1;
for segment = 1:32
  for i = 0:2:8
    i
    load("traj/" + (segment-1) + "_" + i + ".mat");
 
    %{
    n = size(f,2);
    f = fft(f, [], 2);
    k = 0:n-1;
    k(k>n/2) = k(k>n/2) - n;
    f = exp( sx*1j*(loop-1)*k ).*f;
    f = real(ifft( f, [], 2));
    %}
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
  
  nexttile
  enst = [enst,  mean(f(1,:,:).^2, "all")];
  jenst= [jenst, mean(f(2,:,:).^2, "all")];

  plot(enst,jenst);
  hold on
  scatter(enst(end), jenst(end), 'filled', 'markerfacecolor', 'black');
  hold off
  xlim([10, 10.5]);
  ylim([6, 12]);
  xlabel("$\langle \omega^2 \rangle$", "interpreter", "latex");
  ylabel("$\langle j^2 \rangle$", "interpreter", "latex", "rotation", 0);
  title("2D projection");

  drawnow;
  
  % Write each frame to the file.
  currFrame = getframe(gcf);
  writeVideo(vidObj,currFrame);
  end
end
  
%end
% Close the file.
close(vidObj);

