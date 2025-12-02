function visualize_fields2(fields, cmax, display_colorbar)
  %{
  PURPOSE:
  Make a standard visualization of the fields tensor from 
  my MHD simulations. This version makes a single picture
  with magnetic field lines over the vorticity

  INPUT:
  fields - shape [n,n,2] tensor containing curl(u) and curl(B)
  %}
  
  titles = {"$\nabla \times {\bf u}$", "$\nabla \times {\bf B}$"};
  font_size = 32;


      nexttile;
      
      %Draw the field
      imagesc(fields(:,:,1));
        
      j = fft2(fields(:,:,2));
      
      n = size(fields,1);
      k = 0:n-1;
      k(k>n/2) = k(k>n/2) - n;

      %vector potential
      A = j./(k.^2 + k'.^2);
      A(1,1) = 0;
      A = real(ifft2(A));

      %add mean magnetic field
      [x,y] = meshgrid( (0:n-1)/n*2*pi );
      B0y = 1.0;
      B0x = 0.0;
      A = A - B0y * x + B0x * y;
      
      [X,Y] = meshgrid(1:n);

      hold on
      contour(X, Y, A, 30, 'w', 'LineWidth', 0.5, "color", [1,1,1]*0.4);   % 30 lines, black
      hold off

      %Make aspect ratio 1:1
      axis square;

      %Make y point up instead of down
      set(gca, "ydir", "normal");

      %Provide a colorbar
      if display_colorbar
        colorbar();
      end
   
      %Lock the colorbar limits
      clim( [-1,1] * cmax(1) );

      %Turn off ticks
      set(gca, 'XTick', [], 'YTick', []);

      %Add a title for each field
      title(titles{1}, "interpreter", "latex", "color", "w", "fontsize", font_size);
  
      %Set the background color to black
      set(gcf, "color", "k");

      ax = gca;
      ax.XColor = 'k';
      ax.YColor = 'k';
end