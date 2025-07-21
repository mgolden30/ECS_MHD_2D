
clear;
load("../spatial_diag.mat");
load("../fourier_diag.mat");

tiledlayout(2,2);

k = 8;

for i = 1:2  
  nexttile;
  data = squeeze(diag(k,i,:,:)).';
  data(1,1) = 0;
  imagesc(data);
  %surf(data)
  axis square;
  colorbar;

  set(gca, 'ydi', 'normal');
end

for i = 1:2
  nexttile;
  data = squeeze(diag(k,i,:,:)).';
  data(1,1) = 0;

  data = fftshift(fft2(data));
  data = abs(data);

  imagesc(log10(data));
  axis square;
  colorbar;

  set(gca, 'ydi', 'normal');
end

nexttile
nbins = 16;
histogram(diag, nbins);

colormap jet
