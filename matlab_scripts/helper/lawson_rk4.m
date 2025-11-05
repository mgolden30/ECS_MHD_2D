function fields = lawson_rk4( fields, t, steps, params )
  %{
  
  %}
  n = size(fields,1);
  grid = construct_grid(n);
  h = t/steps;

  c = reshape( [params.nu, params.eta], [1,1,2] );
  e = grid.mask .* exp( -h/2*(grid.kx.^2 + grid.ky.^2).*c );
  
  fields = fft2(fields);

  forcing = -4*cos(4*grid.y);

  v = @(fields) explicit_velocity(fields, grid, forcing, params.b0);

  for i = 1:steps
    k1 = h*v(fields);

    fields = fields .* e;
    k1 = k1 .* e;

    k2 = h*v(fields + k1/2);
    k3 = h*v(fields + k2/2);

    fields = fields .* e;
    k1 = k1 .* e;
    k2 = k2 .* e;
    k3 = k3 .* e;

    k4 = h*v(fields + k3);

    fields = fields + (k1 + 2*k2 + 2*k3 + k4) / 6;
  end

  fields = real(ifft2(fields));
end

function f_t = explicit_velocity(f, grid, forcing, b0)
  fx = real(ifft2(1i*grid.kx.*f));
  fy = real(ifft2(1i*grid.ky.*f));
  fu = real(ifft2(grid.to_u.*f));
  fv = real(ifft2(grid.to_v.*f));
  
  %Add the mean magnetic field
  fu(:,:,2) = fu(:,:,2) + b0(1);
  fv(:,:,2) = fv(:,:,2) + b0(2);

  advec = fu.*fx + fv.*fy;

  w_t = -advec(:,:,1) + advec(:,:,2) + forcing;
  w_t = fft2(w_t).* grid.mask;

  %Compute the cross product
  u_times_B = fu(:,:,1).*fv(:,:,2) - fu(:,:,2).*fv(:,:,1);

  k_sq = grid.kx.^2 + grid.ky.^2;
  j_t = fft2(u_times_B) .* k_sq .* grid.mask;

  f_t = cat(3, w_t, j_t);
end

function grid = construct_grid(n)
  k = 0:n-1;
  k(k>n/2) = k(k>n/2) - n;
  k = gpuArray(k);

  kx = reshape(k, [1,n]);
  ky = reshape(k, [n,1]);
  
  to_u = 1i*ky./(kx.^2 + ky.^2);
  to_v =-1i*kx./(kx.^2 + ky.^2);

  %Mask the mean flow
  to_u(1,1) = 0;
  to_v(1,1) = 0;

  mask = (abs(kx) < n/3) & (abs(ky) < n/3);
  mask(1,1) = 0;

  grid.kx = kx;
  grid.ky = ky;

  grid.to_u = to_u;
  grid.to_v = to_v;
  grid.mask = mask;

  [x,y] = meshgrid( (0:n-1)/n*2*pi );
  grid.x = gpuArray(x);
  grid.y = gpuArray(y);
end