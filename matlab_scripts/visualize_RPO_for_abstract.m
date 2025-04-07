clear;

data2 = load("converged_RPO_14721.mat")
data1 = load("jax_rk4.mat");
addpath("derive_adjoint_looping/functions/");


f = data1.f - data2.f;

%{
f = fft( f, [], 2);
f = fft( f, [], 3);
f = abs(f);
f = log10(f);
%}

enst = [];
jenst = [];

%for loop = 1:8

%for i = 1:4:256 
  %i = 0
  %load("traj/" + i + ".mat");
 
  %{
  n = size(f,2);
  f = fft(f, [], 2);
  k = 0:n-1;
  k(k>n/2) = k(k>n/2) - n;
  f = exp( sx*1j*(loop-1)*k ).*f;
  f = real(ifft( f, [], 2));
  %}


  %tiledlayout(1,2);
  %nexttile
  
  clf;
  imagesc(squeeze( f(1,:,:) ).' );
  %title("vorticity");
  axis square;
  %cb = colorbar();
  %set(cb, "XTick", [-10, 0, 10]);
  clim([-10 10]);
  set(gca, 'ydir', 'normal');
  xticks([]); yticks([]);
  saveas( gcf, "pre-vorticity.png" );


  %nexttile
  imagesc(squeeze( f(2,:,:) ).' );
  %title("current");
  axis square;
  %cb = colorbar();
  %set(cb, "XTick", [-10, 0, 10]);
  clim([-10 10]);
  set(gca, 'ydir', 'normal');
  xticks([]); yticks([]);
  saveas( gcf, "pre-current.png" );




load("converged_RPO_14721.mat")
%load("jax_rk4.mat");
addpath("derive_adjoint_looping/functions/");


%{
f = fft( f, [], 2);
f = fft( f, [], 3);
f = abs(f);
f = log10(f);
%}

enst = [];
jenst = [];

%for loop = 1:8

%for i = 1:4:256 
  %i = 0
  %load("traj/" + i + ".mat");
 
  %{
  n = size(f,2);
  f = fft(f, [], 2);
  k = 0:n-1;
  k(k>n/2) = k(k>n/2) - n;
  f = exp( sx*1j*(loop-1)*k ).*f;
  f = real(ifft( f, [], 2));
  %}


  %tiledlayout(1,2);
  %nexttile
  
  clf;
  imagesc(squeeze( f(1,:,:) ).' );
  %title("vorticity");
  axis square;
  %cb = colorbar();
  %set(cb, "XTick", [-10, 0, 10]);
  clim([-10 10]);
  set(gca, 'ydir', 'normal');
  xticks([]); yticks([]);
  saveas( gcf, "vorticity.png" );


  %nexttile
  imagesc(squeeze( f(2,:,:) ).' );
  %title("current");
  axis square;
  %cb = colorbar();
  %set(cb, "XTick", [-10, 0, 10]);
  clim([-10 10]);
  set(gca, 'ydir', 'normal');
  xticks([]); yticks([]);
  saveas( gcf, "current.png" );
