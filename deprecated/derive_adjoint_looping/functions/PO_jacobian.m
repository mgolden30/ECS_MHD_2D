function [Jv] = PO_jacobian( z, dz, params, d )
  %{
  PURPOSE:
  Vector objective function

  INPUT:
  fields - a [n,n,2] tensor containing vorticity and current
  T  - period
  sx - shift in x

  OUTPUT:
  f - our loss function 
      f = norm( R_theta(fields(T)) - fields(0) )
  %}


  n = params.n;

  T  = z(end-1);
  sx = z(end);
  fields = reshape( z(1:end-2), [n,n,2] );

  dT  = dz(end-1);
  dsx = dz(end);
  dfields = reshape( dz(1:end-2), [n,n,2] );

  steps = params.steps;

  dt = T/steps;
  y  = 0*fields; %derivative with respect to T
  fields0  = fields;
  dfields0 = dfields;



  %Create a [n,n,2,b] tensor so we can vectorize the action of the jacobian 
  tangnt = cat(4, dfields0, y);

  %Wrap up evolving fields and dfields into the same routine to maximize
  %speed. 
  fields = fft2(fields);
  tangnt = fft2(tangnt);

  %Step 1: integrate forward with RK4
  for i = 1:steps
    [k1, t1] = state_vel_with_jacobian( fields, tangnt, d, params );
    k1 = dt*k1;
    t1 = dt*t1;
    t1(:,:,:,2) = t1(:,:,:,2) + k1/T;

    [k2, t2] = state_vel_with_jacobian( fields + k1/2, tangnt + t1/2, d, params );
    k2 = dt*k2;
    t2 = dt*t2;
    t2(:,:,:,2) = t2(:,:,:,2) + k2/T;
    
    [k3, t3] = state_vel_with_jacobian( fields + k2/2, tangnt + t2/2, d, params );
    k3 = dt*k3;
    t3 = dt*t3;
    t3(:,:,:,2) = t3(:,:,:,2) + k3/T;
    
    [k4, t4] = state_vel_with_jacobian( fields + k3,   tangnt + t3,   d, params );
    k4 = dt*k4;
    t4 = dt*t4;
    t4(:,:,:,2) = t4(:,:,:,2) + k2/T;
    
  
    fields = fields + (k1 + 2*k2 + 2*k3 + k4)/6;
    tangnt = tangnt + (t1 + 2*t2 + 2*t3 + t4)/6;
  end

  fields = real(ifft2(fields));
  tangnt = real(ifft2(tangnt));

  %commands for translating data
  shift = @(x) real(ifft2( exp(1i*d.kx*sx) .* fft2(x) ));
  shift_deriv = @(x) real(ifft2( 1i*d.kx .* exp(1i*d.kx*sx) .* fft2(x) ));
  deriv = @(x) real(ifft2( 1i*d.kx .* fft2(x) ));

  %Compute difference vector
  df = tangnt(:,:,:,1) + dT*tangnt(:,:,:,2) - shift(dfields0) - shift_deriv(fields0)*dsx;

  %Phase condition 1: orthogonality to time derivative of system
  phase_t = mean( state_vel( fields0, d, params ).*dfields0, "all" );

  %Phase condition 2: orthogonality to translation in x
  phase_x = mean( deriv(fields0).*dfields0, "all" );

  Jv = [df(:); phase_t; phase_x];
end