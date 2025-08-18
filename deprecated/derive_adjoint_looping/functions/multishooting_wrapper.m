function [L, dLdz] = multishooting_wrapper(z, params, d, metric)
  %Wrap up adjoint looping nicely for an optimizer. The optimizer does not
  %care about our physical interpretation of z.


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % unpack the state vector z
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  n = params.n;
  b = params.b;
 
  assert(numel(z) == 2*n*n*b + 2);

  T = z(end-1);
  sx= z(end);
  f = reshape( z(1:end-2), [n,n,2,b] );



  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Evaluate loss and gradient
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  [L, dLdf, dLdT, dLdsx] = adjoint_looping_multishooting( f, T, sx, params, d, metric );



  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % pack the gradient
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  dLdz = [dLdf(:); dLdT; dLdsx];  

end 