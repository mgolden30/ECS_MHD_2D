function   plot_projection(proj, i, closed)
  
  %Close the loop if needed
  if closed
    proj = [proj, proj(:,1)];
  end

  %plot the projection
  plot( proj(1,:), proj(2,:), 'LineWidth', 2, 'color', 'w' );
  hold on
    scatter(proj(1,i), proj(2,i), 'filled');
  hold off

  set(gca, 'color', 'k');
  axis square;
end