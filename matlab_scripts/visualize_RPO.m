%{
This script visualizes an RPO
%}

clear;
addpath("helper/")

%PARAMETERS
filename = "../solutions/Re40/RPO6.npz";
display_colorbar = false;
cmax = [6,6]; %max colorbar for vorticity and current
comoving = true;


%Load data
[fields, T, symmetry, params] = read_npz(filename);

%visualize the initial data
visualize_fields(fields, cmax, display_colorbar);
colormap blueblackred
drawnow;

%Decide how often to plot
ministeps = 32;
assert(mod(params.steps, ministeps) == 0);
snapshots = round(params.steps / ministeps);

n = size(fields,1);
fs = zeros(n,n,2,snapshots);
fs(:,:,:,1) = fields;

fprintf("Generating trajectory...\n");
tic
for i = 1:snapshots-1
    fs(:,:,:,i+1) = lawson_rk4( fs(:,:,:,i), T/snapshots, ministeps, params );
end
walltime = toc;
fprintf("Completed in %.3f seconds...", walltime);

%%

kx = 0:n-1;
kx(kx > n/2) = kx(kx > n/2) -n;
kx = reshape(kx, [1,n]);

sx = symmetry.sx;
if sx > pi
  sx = sx - 2*pi;
end

while(true)
  for i = 1:snapshots-1
    data = fs(:,:,:,i);
    
    if comoving
      data = fft2(data); 
      data = exp( 1i * kx * sx * (i-1)/snapshots ) .* data;
      data = real(ifft2(data));
    end

    visualize_fields(data, cmax, display_colorbar);
    drawnow;
  end
end

return

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

