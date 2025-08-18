%{
While we are ultimately interested in periodic orbits and invariant tori,
simpler structures like equilibria are interesting to partition state
space.
%}

clear;

addpath("functions\");

n = 256;

nu  = 1/100;
eta = 1/100;

[x,y] = meshgrid( (0:(n-1))/n*2*pi );

seed = 0;
fields = random_initial_data( n, seed );

fields(:,:,1) = cos(3*x) + cos(3*y);
fields(:,:,2) = 0*x;

params.n = n;
params.nu  = 1/100;
params.eta = 1/100;
params.force = -4*cos(4*y);

params.Bx0 = 0;
params.By0 = 0.1;

dt = 0.005;

d = deriv_matrices( n );

%Move to GPU
fields = gpuArray(fields);
names = fieldnames(d);
for i = 1:numel(names)
  d.(names{i}) = gpuArray(d.(names{i}));
end

vx = 0; %assume at rest
z = [fields(:); vx];



%% Verify the action of the Jacobian and adjoint
%load("equilibria/6.mat");
addpath("functions/");


%Use f as the tangent vector
f = equilibria_objective(z, d, params);
norm(f)

v = [f; 1];


%Make a random cotangent vector
n = params.n;
seed = 0;
c = random_initial_data(n, seed);
c = c(:);

%Define the Jacobian
J = @(x, flag) matrix_free_jacobian(z, x, flag, d, params);

dot1 = dot( c, J(v, 'notransp') );
dot2 = dot( J(c, 'transp'), v );

relative_error = (dot1 - dot2) / abs(dot1);




%Newton-Krylov
maxit = 10000000;
damp  = 1.0;
inner = 2*1024;
outer = 1;
tol = 1e-6;

for i = 1:maxit
  %Compute the state space velocity
  f = equilibria_objective(z, d, params);
 
  %J = @(dz) equilibria_jacobian(z, dz, d, params);
  %[dz, ~] = gmres( J, f, inner, tol, outer );
  
  J = @(x, flag) matrix_free_jacobian(z, x, flag, d, params);
  %M = @(x, flag) preconditioner(x, flag, d, params);

  tic
  [dz, converge_flag, relres] = lsqr( J, f, tol, inner);
  walltime = toc;

  fprintf("%03d: |f| = %e, walltime = %f, relres = %e\n", i, norm(f), walltime, relres );

  z = z - damp * dz;

  %visualize_fields_extra( reshape(z(1:end-1), [n,n,2]), params, d );
  visualize_fields( reshape(z(1:end-1), [n,n,2]) );
  
  drawnow;
end




function f = equilibria_objective(z, d, params)
  vx = z(end);
  fields = reshape( z(1:end-1), [params.n, params.n, 2] );

  vel = state_vel( fields, d, params);

  dfields_dx = real(ifft2(  1i*d.kx.* fft2(fields)  ));

  vel = vel + vx * dfields_dx;

  %f = [vel(:); 0];
  f = vel(:);
end


function df = equilibria_jacobian(z, dz, d, params)
  vx = z(end);
  fields = reshape( z(1:end-1), [params.n, params.n, 2] );

  dvx = dz(end);
  dfields = reshape( dz(1:end-1), [params.n, params.n, 2] );

  vel = state_vel( fields, d, params);

  dfields_dx = real(ifft2(  1i*d.kx.* fft2(fields)  ));
  ddfields_dx = real(ifft2(  1i*d.kx.* fft2(dfields)  ));

  %vel = vel + vx * dfields_dx;

  Jv = tangent_vector_velocity( fields, dfields, d, params );
  dvel = Jv + dvx * dfields_dx + vx*ddfields_dx;

  %phase condition
  %phase = mean( dfields(:) .* dfields_dx(:) );

  %df = [dvel(:); phase];
  df = dvel(:); 
end




function df = equilibria_jacobian_transpose(z, dz, d, params)
  vx = z(end);
  fields = reshape( z(1:end-1), [params.n, params.n, 2] );

  dfields = reshape( dz(1:end), [params.n, params.n, 2] );

  vel = state_vel( fields, d, params);

  dfields_dx = real(ifft2(  1i*d.kx.* fft2(fields)  ));
  ddfields_dx = real(ifft2(  1i*d.kx.* fft2(dfields)  ));

  vel = vel + vx * dfields_dx;

  JTv = cotangent_vector_velocity( fields, dfields, d, params );
  
  df = [
       JTv(:) - vx*ddfields_dx(:);
       sum( dfields_dx(:) .* dfields(:) );   
       ];

  %df = [dvel(:); phase];
end


function Jx = matrix_free_jacobian(z, x, flag, d, params)
  %{
  To use the lsqr function, we need a function handle with that takes a
  flag for transpose vs no transpose.

  z - our current guess of equilibria
  x - tangent or cotangent vector
  flag - transpose flag
  %}

  if strcmp(flag,'notransp') % Compute A*x
    Jx = equilibria_jacobian(z, x, d, params);
  elseif strcmp(flag,'transp') % Compute A'*x
    Jx = equilibria_jacobian_transpose(z, x, d, params);
  end
end



function Mx = preconditioner(x, flag, d, params)
  %Use a self-adjoint operator so it doesn't matter if trans or notrans
  
  vx = x(end);
  f  = reshape( x(1:end-1), [params.n, params.n, 2] );

  f = real(ifft2( d.inv_lap.^2 .* fft2(f)  ));

  Mx = [f(:); vx];
end