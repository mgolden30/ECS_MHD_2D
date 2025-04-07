function [f_t, df_t] = state_vel_with_jacobian( fields, dfields, d, params )
  %{
  PURPOSE:
  Newton-GMRES requires many evaluations of the Jacobian action of forward
  time evolution. This function aims to combine several function calls to
  reduce walltime.

  INPUT
  fields - [n,n,2] tensor where fields(:,:,1) is vorticity and
           fields(:,:,2) is the current.
  dfields - [n,n,2,b] tensor. b is a batch dimension for batched evaluation
            of matrix-vector product
  
  Assume both of these are passed in Fourier space to reduce the number of
  Fourier transforms we need to compute.
  %}
  
  %Check that we are being passed complex Fourier coefficients
  assert( ~isreal( fields) );
  assert( ~isreal(dfields) );

  %Compute gradients of fields  
  fx = real(ifft2( 1i * d.kx .* fields ));
  fy = real(ifft2( 1i * d.ky .* fields ));

  %uncurl
  fu = real(ifft2( d.to_u .* fields ));
  fv = real(ifft2( d.to_v .* fields ));

  %Add mean field
  fu(:,:,2) = fu(:,:,2) + params.Bx0;
  fv(:,:,2) = fv(:,:,2) + params.By0;

  %Compute "advection" containing hydro and magnetic
  advection = fu.*fx + fv.*fy;

  %Compute vorticity time derivative
  w_t = -advection(:,:,1) + advection(:,:,2) + params.force;
  w_t =  d.mask .* (fft2(w_t) - params.nu * d.k_sq .* fields(:,:,1)); %dealias and add dissipation

  %Compute current time derivative
  cross = fu(:,:,1).*fv(:,:,2) - fv(:,:,1).*fu(:,:,2);
  j_t  = fft2( cross ) - params.eta * fields(:,:,2);
  j_t  = d.mask .* d.k_sq .* j_t;

  %Combine these into a single time derivative
  f_t = cat(3, w_t, j_t);




  %Evaluate the action on a tangent field
  
  %Compute gradients of fields  
  dfx = real(ifft2( 1i * d.kx .* dfields ));
  dfy = real(ifft2( 1i * d.ky .* dfields ));

  %uncurl
  dfu = real(ifft2( d.to_u .* dfields ));
  dfv = real(ifft2( d.to_v .* dfields ));

  %Compute "advection" containing hydro and magnetic
  dadvection = dfu.*fx + dfv.*fy + fu.*dfx + fv.*dfy;

  dw_t = -dadvection(:,:,1,:) + dadvection(:,:,2,:);
  dw_t =  d.mask .* (fft2(dw_t) - params.nu * d.k_sq .* dfields(:,:,1,:)); %dealias and add dissipation

  %Compute current time derivative
  dcross = dfu(:,:,1,:).*fv(:,:,2,:) - dfv(:,:,1,:).*fu(:,:,2,:) + fu(:,:,1,:).*dfv(:,:,2,:) - fv(:,:,1,:).*dfu(:,:,2,:);
  dj_t  = fft2( dcross ) - params.eta * dfields(:,:,2,:);
  dj_t  = d.mask .* d.k_sq .* dj_t;

  %Combine these into a single time derivative
  df_t = cat(3, dw_t, dj_t);
end