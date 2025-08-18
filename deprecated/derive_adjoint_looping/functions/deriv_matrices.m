function d = deriv_matrices( n )
  k= 0:n-1;
  k(k>n/2) = k(k>n/2) - n;

  [kx, ky] = meshgrid( k );

  mask = abs(kx) < n/3 & abs(ky) < n/3;

  to_u = 1i*ky./(kx.^2 + ky.^2).*mask;
  to_v =-1i*kx./(kx.^2 + ky.^2).*mask;

  to_u(1,1) = 0;
  to_v(1,1) = 0;
  
  inv_lap = 1./(kx.^2 + ky.^2);
  inv_lap(1,1) = 0;

  kx = kx.*mask;
  ky = ky.*mask;
  
  d.kx = kx;
  d.ky = ky;
  d.to_u = to_u;
  d.to_v = to_v;

  d.k_sq = kx.^2 + ky.^2;
  d.mask = mask;

  d.inv_lap = inv_lap;
end