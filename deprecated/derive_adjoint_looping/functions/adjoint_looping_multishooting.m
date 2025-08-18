function [L, dLdx, dLdT, dLdsx] = adjoint_looping_multishooting( f, T, sx, params, d, metric )
  %{
  PURPOSE:
  Compute a scalar cost function for Relative Periodic Orbits (RPOs). This
  will use a multishooting scheme to improve convergence.

  INPUT:
  fields - a [n,n,2,b] tenor. b is the batch size, which will be integrated
           in parallel.
  T  - period
  sx - shift in x
  metric - a function handle such that metric(v)
           computes the covector 
           $ g_{ij} v^j $

  OUTPUT:
  f - our loss function 
      f = norm( fields(T) - R_theta(fields(0)) )
  %}

  b = size(f,4);
  n = params.n;
  steps = params.steps; %steps per segment
  
  %In order to evolve backwards in time, we need to save our trajectory
  fs = zeros(n,n,2,b,steps);
  k1s = zeros(n,n,2,b,steps);
  k2s = zeros(n,n,2,b,steps);
  k3s = zeros(n,n,2,b,steps);
  %Only need up to k3

  %timestep is total period per step per segment
  dt = T/steps/b;

 
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Step 1: integrate forward with RK4
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Evolve fields in the Fourier basis to minimize the number of times we have to call fft2 
  f = fft2(f);
  
  %y is shorthand for df/dT, where f is fields. It starts out as zero
  y = 0*f;

  for i = 1:steps
    [k1, y1] = state_vel_with_tangent(f, y, d, params);
    %multiply by timestep. Note this induces a new contribution to y1!
    k1 = dt*k1;
    y1 = dt*y1 + k1/T;
    
    [k2, y2] = state_vel_with_tangent(f + k1/2, y + y1/2, d, params);
    k2 = dt*k2;
    y2 = dt*y2 + k2/T;

    [k3, y3] = state_vel_with_tangent(f + k2/2, y + y2/2, d, params);
    k3 = dt*k3;
    y3 = dt*y3 + k3/T;
  
    [k4, y4] = state_vel_with_tangent(f + k3, y + y3, d, params);
    k4 = dt*k4;
    y4 = dt*y4 + k4/T;
  

    %Save BEFORE you update
    fs( :,:,:,:,i) = f;
    k1s(:,:,:,:,i) = k1;
    k2s(:,:,:,:,i) = k2;
    k3s(:,:,:,:,i) = k3;

    f = f + (k1 + 2*k2 + 2*k3 + k4)/6;
    y = y + (y1 + 2*y2 + 2*y3 + y4)/6;
  end

  %{
  %Debugging
  figure(1);
  non_integrated = real(ifft2( fs(:,:,:,2,1) ));
  integrated     = real(ifft2( f( :,:,:,1) ));
  check_segment( integrated, non_integrated );
  %}

  % commands for translating data
  %shift = @(x) real(ifft2( exp(1i*d.kx*sx/b) .* fft2(x) ));
  unshift = @(x) real(ifft2( exp(-1i*d.kx*sx/b) .* fft2(x) ));
  %deriv = @(x) real(ifft2( 1i*d.kx .* exp(1i*d.kx*sx/b) .* fft2(x) ));

  %Shift f along batch dimension if batch is not 1
  if b ~= 1
    dim = 4; %batch dimension
    cycle = @(x) circshift(x, 1, dim); %function to cycle elements
    f = cycle(f);
    y = cycle(y);
  end

  %Compute spatial shift (per segment)
  shiftx = exp(1i * (sx/b) * d.kx);

  %Compute difference vector. This is a tangent vector as currently defined
  u0 = f - shiftx .* fs(:,:,:,:,1);
  

  %Lower the tangent vector with a metric tensor to produce a covector u
  u = metric(u0);

  %Compute objective function sqrt( u_i u^i )
  u  = real(ifft2(u));
  u0 = real(ifft2(u0));
  
 %{
 figure(2);
 check_u( u )
 
 figure(3);
 check_u( u0 )
 %}
 
  %L_per_segement = squeeze(sum( u0.*u, [1,2,3] ))

  L = sqrt( sum( u0(:).*u(:) ) );

  %Define a temporary variable for derivative with respect to translation
  z = 1i * d.kx .* shiftx .* f(:,:,:,:,1)/b;

  %Back to physical space
  y = real(ifft2(y));
  z = real(ifft2(z));

  %Gradient w.r.t period
  dLdT  =  dot( u(:), y(:) ) / L; 
  dLdsx = -dot( u(:), z(:) ) / L;

  %RK4 constants
  b1 = 1/6;
  b2 = 1/3;
  b3 = 1/3;
  b4 = 1/6;

  a21 = 1/2;
  a32 = 1/2;
  a43 = 1;

  %Function for evaluating the action of 
  jacT = @(x,cotang) cotangent_vector_velocity( x, cotang,  d, params );

  u_initial = u;

  %pullback the cycling of segements (just undo the cycling)
  if ( b ~= 1)
    dim = 4; %batch dimension
    uncycle = @(x) circshift(x, -1, dim); %function to cycle elements
    u = uncycle(u);
  end

  fs  = real(ifft2(fs));
  k1s = real(ifft2(k1s));
  k2s = real(ifft2(k2s));
  k3s = real(ifft2(k3s));
  
  for i = steps:-1:1
    x  = fs(:,:,:,i);
    k1 = k1s(:,:,:,i);
    k2 = k2s(:,:,:,i);
    k3 = k3s(:,:,:,i);
    
    u4 = dt*jacT(x + k3*a43, b4*u);
    u3 = dt*jacT(x + k2*a32, b3*u + u4*a43);
    u2 = dt*jacT(x + k1*a21, b2*u + u3*a32);
    u1 = dt*jacT(x,          b1*u + u2*a21);
  
    u = u + u1 + u2 + u3 + u4;
  end

  dLdx = (u - unshift(u_initial))/L;
end


function check_segment( integrated, non_integrated )
  %{
  Check that we integrated the first segment correctly.
  %}
  

  tiledlayout(3,2);

  nexttile
  imagesc( non_integrated(:,:,1) ); xticks([]); yticks([]); colorbar(); clim([-10 10]); axis square;
  title("\omega(1)")

  nexttile
  imagesc( non_integrated(:,:,2) ); xticks([]); yticks([]); colorbar(); clim([-10 10]); axis square;
  title("j(1)")

  nexttile
  imagesc( integrated(:,:,1) ); xticks([]); yticks([]); colorbar(); clim([-10 10]); axis square;
  title("evolved \omega(0)")

  nexttile
  imagesc( integrated(:,:,2) ); xticks([]); yticks([]); colorbar(); clim([-10 10]); axis square;
  title("evolved j(0)")

  nexttile
  imagesc( integrated(:,:,1) - non_integrated(:,:,1) ); xticks([]); yticks([]); colorbar(); clim([-10 10]); axis square;
  title("diff \omega")

  nexttile
  imagesc( integrated(:,:,2) - non_integrated(:,:,2) ); xticks([]); yticks([]); colorbar(); clim([-10 10]); axis square;
  title("diff j")

end


function check_u( u )
  %{
  Check that we integrated the first segment correctly.
  %}
  
  b = size(u,4);

  tiledlayout(b,2);

  for i = 1:b

  nexttile
  imagesc( u(:,:,1,i) ); xticks([]); yticks([]); colorbar(); clim([-10 10]); axis square;
  
  nexttile
  imagesc( u(:,:,2,i)); xticks([]); yticks([]); colorbar(); clim([-10 10]); axis square;
  end
end