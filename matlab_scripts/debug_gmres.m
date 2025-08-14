clear;

load("../gmres_debug_0.mat");

b = zeros(size(H,1),1);
b(1) = 1;


%Ensure b is a column vector
b = reshape(b, [], 1);
S = svd(H);

figure(1);

tiledlayout(2,2);

nexttile
semilogy(S ,'o');
title("singular values");
xlabel("n")
ylabel("\sigma_n");
axis square;

nexttile
rel_res = trust_sweep( H, b );
semilogy(rel_res, 'o');
title("pseudoinverse residual");
xlabel("n (trusted)")
ylabel("relative residual");
axis square;

nexttile;

D = Q.'*Q;
imagesc(log10(abs(D)));
colorbar();
axis square;
title("log10(Q^T Q)")


figure(2);

index = 2;

q = Q(2:end-1,index);
%q = Q(1:end-2, index);
%q = Q(3:end, index);
n = 512;
q = reshape( q, n, n, 2  );


tiledlayout(2,2);

nexttile
imagesc( squeeze(q(:,:,1)) )
surf( log10(abs(fftshift(fft2(squeeze(q(:,:,1)))))) )
%imagesc( log10(abs(fftshift(fft2(squeeze(q(:,:,1)))))) )
shading interp
%zlim([-4 3]);
title("GS vector (vorticity)")
axis square
colorbar();
%clim(zlim);
colormap jet

nexttile
imagesc( squeeze(q(:,:,2)) )
surf( log10(abs(fftshift(fft2(squeeze(q(:,:,2)))))) )
%imagesc( log10(abs(fftshift(fft2(squeeze(q(:,:,2)))))) )

shading interp
title("GS vector (current)")
axis square
colorbar();

%Solve for the newton step and visualize it
e1 = zeros(size(H,1),1);
e1(1) = 1;

y = H\e1;
s = Q(:,1:end-1)*y;

s = s(2:end-1);
s = reshape( s, n, n, 2  );

nexttile
imagesc( squeeze(s(:,:,1)) )
title("Newton step (vorticity)")
axis square
colorbar();

nexttile
imagesc( squeeze(s(:,:,2)) )
title("Newton step (current)")
axis square;
colorbar();

function rel_res = trust_sweep( H, b )
  [U, S, V] = svd(H, "econ");
  
  b = reshape(b, [], 1);

  s = diag(S);
  b2 = U.' * b;

  rel_res = [];
  for i = 1:numel(s)
    s_inv = 1./s;
    s_inv(i+1:end) = 1; %untrust these 

    x = s_inv .* b2;
    x = V*x;

    size(x)
    rel_res(i) = norm(H*x - b)/norm(b);
  end
end