%{
Derive adjoint form of MHD equations
%}

clear;

addpath("functions\");

%%
n = 128;

nu  = 1/100;
eta = 1/100;

[x,y] = meshgrid( (0:(n-1))/n*2*pi );

seed = 0;
fields = random_initial_data( n, seed );

params.nu  = 1/100;
params.eta = 1/100;
params.force = -4*cos(4*y);

params.Bx0 = 0;
params.By0 = 0.1;

dt = 0.01;

d = deriv_matrices( n );

for i = 1:1024
  k1 = dt*state_vel( fields, d, params );
  k2 = dt*state_vel( fields + k1/2, d, params );
  k3 = dt*state_vel( fields + k2/2, d, params );
  k4 = dt*state_vel( fields + k3,   d, params );
    
  fields = fields + ( k1 + 2* k2 + 2* k3 +  k4)/6;
  
  if mod(i, 16) ~= 0
    continue;
  end

  tiledlayout(1,2);

  nexttile
  imagesc(fields(:,:,1)); axis square; colorbar();
  nexttile
  imagesc(fields(:,:,2)); axis square; colorbar(); 
  drawnow;
end


%% Use this realistic field to 
tang = random_initial_data( n, 1 );
cotang = random_initial_data(n, 2);

dot0 = mean( tang.*cotang, "all" );

Jtang   =   tangent_vector_velocity(fields,   tang, d, params);
Jcotang = cotangent_vector_velocity(fields, cotang, d, params);

dot1 = mean( cotang.*Jtang, "all" );
dot2 = mean( Jcotang.*tang, "all" );

fprintf("\n");
fprintf("<u| v> = %.12f\n", dot0);
fprintf("<u|Jv> = %.12f\n", dot1);
fprintf("<Ju|v> = %.12f\n", dot2);

%%

params.n = n;
params.steps = 512;
T = 5;

%[f, dfdT, dfdx] = adjoint_looping( fields, T, params, d );

z = [fields(:); T];

maxit = 8;

lr = 1e-3;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-6;

loss_fn = @(z) wrapper(z, params, d);
[z, fs] = adam_optimization( loss_fn, z, lr, maxit, beta1, beta2, epsilon);

semilogy(fs);


%% Check that adding a metric didn't break everything

load("candidates/candidate_3.mat")

%The metric lowers the index
metric = @(v) v; %Euclidean metric








function [f, df] = wrapper(z, params, d)
  T = z(end);
  n = params.n;
  fields = reshape(z(1:end-1), [n,n,2]);

  [f, dfdT, dfdx] = adjoint_looping( fields, T, params, d );

  df = [dfdx(:); dfdT];
end