%% Newton-Krylov iteration
clear;
load("candidates/candidate_3.mat");

maxit = 32;
inner = 16;
outer = 1;
tol   = 1e-6;
damp  = 0.01;

%{
  tic
  f = PO_objective(z, params, d);
  toc
 
  figure(1)
  visualize_fields( reshape(f(1:end-2), [params.n,params.n,2]) );
  drawnow;

%  params.steps = 1;
%  z(end-1) = 0.005;

  J = @(dz) PO_jacobian(z, dz, params, d);

  tic
  [fields, Jf] = PO_jacobian(z, f, params, d);
  toc

  %Jf = J(f);
 
  figure(2)
  visualize_fields( reshape(Jf(1:end-2), [params.n,params.n,2]) );

  
  figure(3)
  visualize_fields( fields );

  drawnow;
  return;
%}


for i = 1:maxit
  tic
  f = PO_objective(z, params, d);
  toc
  
  visualize_fields( reshape(f(1:end-2), [params.n,params.n,2]) );
  drawnow;

  J = @(dz) PO_jacobian(z, dz, params, d);
  M = @(z) precond(z, params, d);

  tic;
  [dz, flag, residual] = gmres( J, f, inner, tol, outer, M );
  walltime = toc;

  z = z - damp*dz;

  fprintf("%03d: |f| = %e, GMRES residual = %e, walltime = %.3f\n", i, norm(f), residual, walltime );
end

save("z_GMRES.mat", "z", "params", "d");



function Mz = precond( z, params, d )
  %Preconditioner for GMRES
  n = params.n;
  fields = reshape(z(1:end-2), [n,n,2]);

  fields = real(ifft2( d.inv_lap .* fft2(fields) ));

  Mz = [fields(:); z(end-1:end) ];
end
