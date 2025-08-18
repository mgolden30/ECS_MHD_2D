function [dfdt] = tangent_vector_velocity( fields, tang,  d, params )
  w = fft2(fields(:,:,1));
  j = fft2(fields(:,:,2));

  tw = fft2(tang(:,:,1));
  tj = fft2(tang(:,:,2));

  wx = real(ifft2( 1i * d.kx .* w ));
  wy = real(ifft2( 1i * d.ky .* w ));
  jx = real(ifft2( 1i * d.kx .* j ));
  jy = real(ifft2( 1i * d.ky .* j ));
  
  u = real(ifft2( d.to_u .* w ));
  v = real(ifft2( d.to_v .* w ));
  a = real(ifft2( d.to_u .* j )) + params.Bx0;
  b = real(ifft2( d.to_v .* j )) + params.By0;

  twx = real(ifft2( 1i * d.kx .* tw ));
  twy = real(ifft2( 1i * d.ky .* tw ));
  tjx = real(ifft2( 1i * d.kx .* tj ));
  tjy = real(ifft2( 1i * d.ky .* tj ));
  
  tu = real(ifft2( d.to_u .* tw ));
  tv = real(ifft2( d.to_v .* tw ));
  ta = real(ifft2( d.to_u .* tj ));
  tb = real(ifft2( d.to_v .* tj ));

  lap_tw =  real(ifft2( d.k_sq .* tw ));
  lap_tj =  real(ifft2( d.k_sq .* tj ));

  %Compute dynamics of tw
  dwdt = - u.*twx - v.*twy - tu.*wx - tv.*wy ...
         + a.*tjx + b.*tjy + ta.*jx + tb.*jy ...
         - params.nu*lap_tw;

  %Compute dynamics of tj
  djdt = real(ifft2( d.k_sq.*fft2(tu.*b - tv.*a + u.*tb - v.*ta) )) - params.eta*lap_tj;

  %Dealias and stack
  dwdt = real(ifft2( d.mask .* fft2(dwdt) ));
  djdt = real(ifft2( d.mask .* fft2(djdt) ));
  dfdt = cat(3, dwdt, djdt);
end