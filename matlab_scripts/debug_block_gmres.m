data = [
1, 2.771707057952881
2, 3.035787343978882
3, 3.1741907596588135
4, 3.3219053745269775
5, 3.4201667308807373
6, 3.7718915939331055
7, 3.8322010040283203
8, 4.3182947635650635
9, 4.468538045883179
10, 4.664758205413818
11, 4.796645641326904
12, 5.163766622543335
13, 5.311567068099976
14, 5.530094385147095
15, 6.123265027999878];

scatter( data(:,1), data(:,2)/data(1,2) );
xlabel("batch dimension");
ylabel("walltime (arb)");

xticks(1:2:20);
yline(2, '--')

%%
clear
load('bgmres.mat');


figure(1);
tiledlayout(2,2);

nexttile
imagesc(Q);
clim([-1 1] * mean(abs(Q), "all"));
title('Q');

nexttile
imagesc( log10(abs(Q.'*Q)) );
colorbar();
clim([-12 0]);
axis square;
title('log_{10} |Q^T Q|')

nexttile
imagesc(log10(abs(H)));
colorbar();
title("log_{10} |H|")


nexttile
semilogy( svd(H));
yline(1);



figure(2);

index = 60;

f = Q(2:end-1, index);

n = 128;
f = reshape(f, [n,n,2]);


tiledlayout(2,2);

nexttile
imagesc( squeeze(f(:,:,1)).' );
axis square;

nexttile
imagesc( squeeze(f(:,:,2)).' );
axis square;

nexttile
imagesc( abs(fftshift(fft2(f(:,:,1)))) )
