function [f, dfdx, dfdT, dfdsx] = adjoint_looping( fields, T, sx, params, d, metric )
  %{
  PURPOSE:
  Compute a scalar cost function for Relative Periodic Orbits (RPOs).

  INPUT:
  fields - a [n,n,2] tensor containing vorticity and current
  T  - period
  sx - shift in x
  metric - a function handle such that metric(v)
           computes the covector 
           $ g_{ij} v^j $


  OUTPUT:
  f - our loss function 
      f = norm( fields(T) - R_theta(fields(0)) )
  %}

  n = params.n;
  steps = params.steps;
  
  %In order to evolve backwards in time, we need to save our trajectory
  traj = zeros(n,n,2,steps);
  k1s = zeros(n,n,2,steps);
  k2s = zeros(n,n,2,steps);
  k3s = zeros(n,n,2,steps);
  %Only need up to k3

  dt = T/steps;

  y = 0*fields;

  %Step 1: integrate forward with RK4
  for i = 1:steps
  
    k1 = dt*state_vel( fields, d, params );
    y1 = dt*tangent_vector_velocity(fields, y, d, params) + k1/T;

    k2 = dt*state_vel( fields + k1/2, d, params );
    y2 = dt*tangent_vector_velocity(fields + k1/2, y + y1/2, d, params) + k2/T;

    k3 = dt*state_vel( fields + k2/2, d, params );
    y3 = dt*tangent_vector_velocity(fields + k2/2, y + y2/2, d, params) + k3/T;

    k4 = dt*state_vel( fields + k3, d, params );
    y4 = dt*tangent_vector_velocity(fields + k3, y + y3, d, params) + k4/T;
  
    %Save before you update
    traj(:,:,:,i) = fields;
    k1s(:,:,:,i)  = k1;
    k2s(:,:,:,i)  = k2;
    k3s(:,:,:,i)  = k3;

    fields = fields + (k1 + 2*k2 + 2*k3 + k4)/6;
    y = y + (y1 + 2*y2 + 2*y3 + y4)/6;
  end

  %commands for translating data
  shift = @(x) real(ifft2( exp(1i*d.kx*sx) .* fft2(x) ));
  unshift = @(x) real(ifft2( exp(-1i*d.kx*sx) .* fft2(x) ));
  deriv = @(x) real(ifft2( 1i*d.kx .* exp(1i*d.kx*sx) .* fft2(x) ));

  %Compute difference vector. There is an upper index and lower index
  %vector that we need to keep track of.
  u0_upper = fields - shift(traj(:,:,:,1));
  u0_lower = metric(u0_upper);

  %Compute objective function
  f = sqrt( sum( u0_upper(:).*u0_lower(:) ) );

  temp = deriv( traj(:,:,:,1) );

  %Gradient w.r.t period
  dfdT  =  dot( u0_lower(:), y(:) ) / f; 
  dfdsx = -dot( u0_lower(:), temp(:) ) / f;

  %RK4 constants
  b1 = 1/6;
  b2 = 1/3;
  b3 = 1/3;
  b4 = 1/6;

  a21 = 1/2;
  a32 = 1/2;
  a43 = 1;

  %Pullback the covector
  u = u0_lower;

  %vel  = @(x) state_vel( x, d, params );
  jacT = @(x,cotang) cotangent_vector_velocity( x, cotang,  d, params );

  for i = steps:-1:1
    x  = traj(:,:,:,i);
    k1 = k1s(:,:,:,i);
    k2 = k2s(:,:,:,i);
    k3 = k3s(:,:,:,i);
    
    u4 = dt*jacT(x + k3*a43, b4*u);
    u3 = dt*jacT(x + k2*a32, b3*u + u4*a43);
    u2 = dt*jacT(x + k1*a21, b2*u + u3*a32);
    u1 = dt*jacT(x,          b1*u + u2*a21);
  
    u = u + u1 + u2 + u3 + u4;
  end

  dfdx = (u - unshift(u0_lower))/f;
end