function [dfdt] = cotangent_vector_velocity( fields, cotang,  d, params )
  %{
  Implement the cotangent action
  %}
  
  w = fft2(fields(:,:,1));
  j = fft2(fields(:,:,2));

  cw = fft2(cotang(:,:,1));
  cj = fft2(cotang(:,:,2));

  wx = real(ifft2( 1i * d.kx .* w ));
  wy = real(ifft2( 1i * d.ky .* w ));
  jx = real(ifft2( 1i * d.kx .* j ));
  jy = real(ifft2( 1i * d.ky .* j ));
  
  u = real(ifft2( d.to_u .* w ));
  v = real(ifft2( d.to_v .* w ));
  a = real(ifft2( d.to_u .* j )) + params.Bx0;
  b = real(ifft2( d.to_v .* j )) + params.By0;

  cwx = real(ifft2( 1i * d.kx .* cw ));
  cwy = real(ifft2( 1i * d.ky .* cw ));
  %cjx = real(ifft2( 1i * d.kx .* cj ));
  %cjy = real(ifft2( 1i * d.ky .* cj ));
  
  %cu = real(ifft2( d.to_u .* cw ));
  %cv = real(ifft2( d.to_v .* cw ));
  %ca = real(ifft2( d.to_u .* cj ));
  %cb = real(ifft2( d.to_v .* cj ));

  lap_cw =  real(ifft2( d.k_sq .* cw ));
  lap_cj =  real(ifft2( d.k_sq .* cj ));


  %define macros so I can think
  inv_lap = @(f) real(ifft2( d.inv_lap .* fft2(f)  ));
  lap  = @(w) real(ifft2( -d.k_sq .* fft2(w) ));
  to_u = @(w) real(ifft2(  d.to_u .* fft2(w) ));
  to_v = @(w) real(ifft2(  d.to_v .* fft2(w) ));

  %cw = real(ifft2(cw));
  cj = real(ifft2(cj));

  %Compute dynamics of cw
  dwdt = u.*cwx + v.*cwy + inv_lap(wx.*cwy - wy.*cwx) ...
       + to_u( lap(cj).*b ) - to_v(lap(cj).*a) ...
       - params.nu * lap_cw;
       
  %Compute dynamics of tj
  djdt = - a.*cwx - b.*cwy - inv_lap(jx.*cwy - jy.*cwx) ...
         - to_u( lap(cj).*v ) + to_v(lap(cj).*u) ...
         - params.eta * lap_cj;
 
  %Dealias and stack
  dwdt = real(ifft2( d.mask .* fft2(dwdt) ));
  djdt = real(ifft2( d.mask .* fft2(djdt) ));
  dfdt = cat(3, dwdt, djdt);
end