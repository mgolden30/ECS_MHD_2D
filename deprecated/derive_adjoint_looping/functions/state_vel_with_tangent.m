function [f_dot, df_dot] = state_vel_with_tangent( f, df, d, params )
  %{
  Upgraded velocity function to handle batched input.
  fields  - size [n,n,2,b]
  dfields - same as fields. A tangent vector evolve linearly.

  Assume the fields are passed in fourier space.

  OUTPUT:
  f_dot  - velcoity of our fields
  df_dot - velocity of the tangent vector
  %}

  %Check that a Fourier transform has already been applied
  assert( ~isreal(f) );
  %assert( ~isreal(df));

  %Gradients
  fx = real(ifft2( 1i * d.kx .* f ));
  fy = real(ifft2( 1i * d.ky .* f ));
  
  %uncurled vector components u and B
  fu = real(ifft2( d.to_u .* f ));
  fv = real(ifft2( d.to_v .* f ));
  
  %Add mean magnetic field
  fu(:,:,2,:) = fu(:,:,2,:) + params.Bx0;
  fv(:,:,2,:) = fv(:,:,2,:) + params.By0;

  %Gradients
  dfx = real(ifft2( 1i * d.kx .* df ));
  dfy = real(ifft2( 1i * d.ky .* df ));
  
  %uncurled vector components u and B
  dfu = real(ifft2( d.to_u .* df ));
  dfv = real(ifft2( d.to_v .* df ));


  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Dynamics of vorticity
  %%%%%%%%%%%%%%%%%%%%%%%%%%%

  %compute "advection" u dot grad omega and B dot grad j
  advection  = fu.*fx + fv.*fy;
  dadvection = dfu.*fx + dfv.*fy + fu.*dfx + fv.*dfy;
  
  %Evaluate advection and forcing in real space
  w_dot = - advection(:,:,1,:) +  advection(:,:,2,:) + params.force;
  dw_dot= -dadvection(:,:,1,:) + dadvection(:,:,2,:);

  %Transform back to Fourier, mask, add dissipation
  w_dot =  d.mask .* fft2( w_dot) - params.nu * d.k_sq .* f(:,:,1,:);
  dw_dot=  d.mask .* fft2(dw_dot) - params.nu * d.k_sq .*df(:,:,1,:);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Dynamics of current
  %%%%%%%%%%%%%%%%%%%%%%%%%%%

  %Cross product of u and B
  u_times_B = fu(:,:,1,:).*fv(:,:,2,:) - fv(:,:,1,:).*fu(:,:,2,:);
  du_times_B= dfu(:,:,1,:).*fv(:,:,2,:) - dfv(:,:,1,:).*fu(:,:,2,:) + fu(:,:,1,:).*dfv(:,:,2,:) - fv(:,:,1,:).*dfu(:,:,2,:);

  %d_sq contains mask already
  j_dot =  d.k_sq.*( fft2( u_times_B) - params.eta * f(:,:,2,:) );
  dj_dot=  d.k_sq.*( fft2(du_times_B) - params.eta *df(:,:,2,:) );
  
  %Stack both of these into a state velocity
  f_dot = cat(3,  w_dot,  j_dot);
  df_dot= cat(3, dw_dot, dj_dot);
end