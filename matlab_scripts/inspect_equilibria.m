%{
Load and inspect equilibria 
%}

clear;
load( "../solutions/Re100/EQ2_new.mat" );
load( "../solutions/Re100/EQ7.mat" );


plt = @(x, i) imagesc( repmat( squeeze(x(i,:,:)).', [1,1]) ) ;

tiledlayout(2,2);

nexttile
plt(f,1)
axis square; colorbar();
clim([-10 10]);

nexttile
plt(f,2);
axis square; colorbar();
clim([-10 10]);

nexttile
w = squeeze(f(1,:,:));
w = fftshift(fft2(w));
w = abs(w);

surf(log10(w));
shading interp;