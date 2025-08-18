function visualize_fields_extra( fields, params, d )
  u = real(ifft2( d.to_u .* fft2(fields(:,:,1)) ));
  v = real(ifft2( d.to_v .* fft2(fields(:,:,1)) ));

  a = real(ifft2( d.to_u .* fft2(fields(:,:,2)) )) + params.Bx0;
  b = real(ifft2( d.to_v .* fft2(fields(:,:,2)) )) + params.By0;

  n = params.n;

  x = 1:n;
  s = 8;
  mask = mod(x-1, 8) == 0;

  u = u.*mask.*mask.';
  v = v.*mask.*mask.';
  %a = a.*mask.*mask.';
  %b = b.*mask.*mask.';
  
  scale = s;

  tiledlayout(1,2); 
  nexttile
  imagesc(fields(:,:,1)); 
  hold on
  quiver( u, v, scale, "linewidth", 1, "color", "black");
  hold off
  axis square; 
  colorbar(); 
  clim([-10 10]);
  xticks([]);
  yticks([]);
  set(gca, 'ydir', 'normal');
  title("${\bf u}$ and $\nabla \times {\bf u}$", "interpreter", "latex")



  nexttile
  imagesc(fields(:,:,2)); 
  hold on
  %  quiver( a, b, scale, "linewidth", 1, "color", "black");
  [startx, starty] = meshgrid(1:s:n, 1:s:n);
  [x, y] = meshgrid(1:n, 1:n);
  step = 0.1;
  maxvert = 50;
  verts = stream2(x, y, a, b, startx, starty, [step, maxvert] );
  lineobj = streamline(verts);
  set(lineobj, "color", "black");

  verts = stream2(x, y, -a, -b, startx, starty, [step, maxvert] );
  lineobj = streamline(verts);
  set(lineobj, "color", "black");
  

  hold off
  axis square; 
  colorbar(); 
  clim([-10 10]);
  xticks([]);
  yticks([]);
  set(gca, 'ydir', 'normal');
  title("${\bf B}$ lines and $\nabla \times {\bf B}$", "interpreter", "latex");

  colormap bluewhitered;
end