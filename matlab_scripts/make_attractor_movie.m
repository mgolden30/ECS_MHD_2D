%{
visualize attractor
%}

clear;
load("../attractor.mat");

set(gcf, "color", "black");
set(gca, "color", "black");


vidObj = VideoWriter('attractor.avi');
vidObj.FrameRate = 15;
open(vidObj);

ms = 20;
for i = 1:32:size(observables,1)
  x = observables(i,:,:);
  x = squeeze(x);
  scatter( x(:,1), x(:,2), ms, 'o', 'filled', 'markerfacecolor', 'white' );
  R = 20;
  xlim([0, 1]*R);
  ylim([0, 1]*R);
  set(gca, 'color', 'k')
  set(0, 'DefaultAxesXColor', 'w')        % X axis in white
  set(0, 'DefaultAxesYColor', 'w')   
  axis square;

  xlabel("$\langle (\nabla \times u)^2 \rangle$", "interpreter", "latex", "fontsize", 32);
  ylabel("$\langle (\nabla \times B)^2 \rangle$", "interpreter", "latex", "fontsize", 32);

  xticks([0:10:20]);
  yticks([0:10:20]);

  dt = 1/256;
  title( "$t = " + sprintf("%.2f", dt*i) + "$", "interpreter", "latex", "FontSize", 32, "color", "w" )

  box on
  drawnow;
  
  % Write each frame to the file. 
  currFrame = getframe(gcf);
  writeVideo(vidObj,currFrame);
end
  
% Close the file.
close(vidObj);
  