%{
This script visualizes an RPO
%}

clear;
addpath("helper/")

%PARAMETERS
%filename = "../solutions/Re40/RPO6.npz";
filename = "1.npz";
display_colorbar = false;
cmax = 2*[10,10]; %max colorbar for vorticity and current
comoving = false;


%Load data
[fields, T, symmetry, params] = read_npz(filename);

%visualize the initial data
visualize_fields(fields, cmax, display_colorbar);
colormap blueblackred
drawnow;

%Decide how often to plot
ministeps = 8;
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
fprintf("Completed in %.3f seconds...\n", walltime);

%%

%Compute a 2D projection
proj = squeeze(mean(fs.^2, [1,2]));

kx = 0:n-1;
kx(kx > n/2) = kx(kx > n/2) -n;
kx = reshape(kx, [1,n]);
ky = reshape(kx, [n,1]);

sx = symmetry.sx;
if sx > pi
  sx = sx - 2*pi;
end



while(true)
  for i = 1:4:snapshots
    data = fs(:,:,:,i);
    
    if comoving
      data = fft2(data); 
      data = exp( 1i * kx * sx * (i-1)/snapshots ) .* data;
      data = real(ifft2(data));
    end

    tiledlayout(1,2);
    visualize_fields2(data, cmax, display_colorbar);

    nexttile
    closed = false;
    plot_projection(proj, i, closed);
    font_size = 24;
    title("2D projection", "FontSize", font_size);
    drawnow;
  end
end