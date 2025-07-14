%{
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
%}


clear

iteration = 3;
load( "../debug/bgmres_" + iteration + ".mat");
load( "../debug/f_vec_" + iteration + ".mat");


figure(1);
tiledlayout(2,2);

nexttile
imagesc(Q);
clim([-1 1] * mean(abs(Q), "all"));
title('Q');

nexttile
imagesc( log10(abs(Q.'*Q)) );
colorbar();
clim([-20 0]);
axis square;
title('log_{10} |Q^T Q|')

nexttile
imagesc(log10(abs(H)));
colorbar();
title("log_{10} |H|")


nexttile
semilogy( svd(H));
yline(1);

%%
figure(2)
plot_GMRES_residual( Q, H, b );


figure(3);
index = 8;
f = Q(2:end-1, index);
vis_vector(f)



function plot_GMRES_residual( Q, H, b )
  %{
  Make some plots to determine how well block gmres did at solving the
  linear system.
  %}
  
  tiledlayout(1,2);

  nexttile
  b2 = Q.'*b.';
  plot(b2, 'o');
  title("Q^T b");



  %Find block size automatically
  s = max(find( abs(H(:,1)) > 1e-10 )) - 1

  %Number of block evaluations
  m = size(H,2)/s

  res = zeros(m,1);
  for i = 1:m
    H_sub = H(1:s*(i+1), i:s*i);  
    b_sub = b2(1:size(H_sub,1));

    %x_sub = H_sub \ b_sub;

    [U, S, V] = svd( H_sub, "econ");
    S = diag(S);
    S_inv = 1./S;
    S_inv( S < 1.0 ) = 1; %conservative inverse
    
    x_sub = U.' * b_sub;
    x_sub = S_inv .* x_sub;
    x_sub = V * x_sub;
    
    x_sub = pinv(H_sub) * b_sub;

    %maxit = 128;
    %x_sub = lsqr( H_sub, b_sub, 1e-10, maxit);

    %plot(b_sub);
    %pause(1);

    res(i) = norm( H_sub*x_sub - b_sub ) / norm(b_sub);
  end

  nexttile
  semilogy( s*(1:m), res, 'o' );
end



function vis_vector(f)
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
  imagesc( abs(fftshift(fft2(f(:,:,1)))))
end