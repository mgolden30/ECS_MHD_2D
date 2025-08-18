function [dfdt] = state_vel( fields, d, params )
  %{
  Upgraded velocity function to handle batched input.
  fields - size [n,n,2,b]

  Assume the fields are passed in real
  %}

  %Check that a Fourier transform has already been applied
  %assert( ~isreal(fields) );
  fields = fft2(fields);

  %Gradients
  fx = real(ifft2( 1i * d.kx .* fields ));
  fy = real(ifft2( 1i * d.ky .* fields ));
  
  %uncurled vector components u and B
  fu = real(ifft2( d.to_u .* fields ));
  fv = real(ifft2( d.to_v .* fields ));
  
  %Add mean magnetic field
  fu(:,:,2,:) = fu(:,:,2,:) + params.Bx0;
  fv(:,:,2,:) = fv(:,:,2,:) + params.By0;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Dynamics of vorticity
  %%%%%%%%%%%%%%%%%%%%%%%%%%%

  %compute "advection" u dot grad omega and B dot grad j
  advection = fu.*fx + fv.*fy;

  %Evaluate advection and forcing in real space
  dwdt = -advection(:,:,1,:) + advection(:,:,2,:) + params.force;

  %Transform back to Fourier, mask, add dissipation
  dwdt =  d.mask .* fft2(dwdt) - params.nu * d.k_sq .* fields(:,:,1,:);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Dynamics of current
  %%%%%%%%%%%%%%%%%%%%%%%%%%%

  %Cross product of u and B
  u_times_B = fu(:,:,1,:).*fv(:,:,2,:) - fv(:,:,1,:).*fu(:,:,2,:);

  %d_sq contains mask already
  djdt =  d.k_sq.*( fft2(u_times_B) - params.eta * fields(:,:,2,:) );
  
  %Stack both of these into a state velocity
  dfdt = cat(3, dwdt, djdt);

  dfdt = real(ifft2(dfdt));
end