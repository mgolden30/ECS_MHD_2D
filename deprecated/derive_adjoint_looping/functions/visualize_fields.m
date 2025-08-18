function visualize_fields( fields)
  tiledlayout(1,2); 
  nexttile
  imagesc(fields(:,:,1)); axis square; colorbar(); clim([-10 10]);
  set(gca, 'ydir', 'normal');

  nexttile
  imagesc(fields(:,:,2)); axis square; colorbar(); clim([-10 10]);
  set(gca, 'ydir', 'normal');
  
  colormap bluewhitered;
end