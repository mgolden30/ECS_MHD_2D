function [f] = PO_objective( z, params, d )
  %{
  PURPOSE:
  Vector objective function

  INPUT:
  fields - a [n,n,2] tensor containing vorticity and current
  T  - period
  sx - shift in x

  OUTPUT:
  f - our loss function 
      f = norm( fields(T) - R_theta(fields(0)) )
  %}


  n = params.n;

  T  = z(end-1);
  sx = z(end);
  fields = reshape( z(1:end-2), [n,n,2] );

  steps = params.steps;

  dt = T/steps;
  fields0 = fields;

  %Step 1: integrate forward with RK4
  for i = 1:steps
    k1 = dt*state_vel( fields, d, params );
    k2 = dt*state_vel( fields + k1/2, d, params );
    k3 = dt*state_vel( fields + k2/2, d, params );
    k4 = dt*state_vel( fields + k3, d, params );
  
    fields = fields + (k1 + 2*k2 + 2*k3 + k4)/6;
  end

  %commands for translating data
  shift = @(x) real(ifft2( exp(1i*d.kx*sx) .* fft2(x) ));
  
  %Compute difference vector
  f = fields - shift(fields0);
  f = [f(:); 0; 0];
end