function [fields, params] = generate_multishoting_guess( f, T, b, d, params, total_steps )
  n = params.n;
  fields = zeros(n,n,2,b);
  fields(:,:,:,1) = f;

  %Evolve the fields explicitly
  steps_per_segment = ceil( total_steps/b );
  params.steps = steps_per_segment;
  
  f = fields(:,:,:,1);
  dt = T / steps_per_segment / b;

  t = 0;
  for i = 2:b
    %Explicitly re-integrate
    tic
    for j = 1:steps_per_segment
      k1 = dt*state_vel( f,        d, params );
      k2 = dt*state_vel( f + k1/2, d, params );
      k3 = dt*state_vel( f + k2/2, d, params );
      k4 = dt*state_vel( f + k3,   d, params );
    
      f = f + ( k1 + 2* k2 + 2* k3 +  k4)/6;
      t = t + dt;
    end
    toc
    fields(:,:,:,i) = f;
  end
  t
end