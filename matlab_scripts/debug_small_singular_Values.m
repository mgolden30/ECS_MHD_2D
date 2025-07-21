clear;

%load("../gmres_debug_0.mat");
load("../debug_adjoint_GMRES.mat");
H = B;

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

D = U.'*U;
imagesc(log10(abs(D)));
colorbar();
axis square;

nexttile;

D = V.'*V;
imagesc(log10(abs(D)));
colorbar();
axis square;


figure(2);

index = 1;

u = U(:,index);
n = 256;
u = reshape( u, n, n, 2  );

v = V(2:end-1,index);
n = 256;
v = reshape( v, n, n, 2 );


tiledlayout(2,2);

nexttile
imagesc( squeeze(u(:,:,1)) )
title("left vector (vorticity)")
axis square

nexttile
imagesc( squeeze(u(:,:,2)) )
title("left vector (current)")
axis square

nexttile
imagesc( squeeze(v(:,:,1)) )
title("right vector (vorticity)")
axis square

nexttile
imagesc( squeeze(v(:,:,2)) )
title("right vector (current)")
axis square


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