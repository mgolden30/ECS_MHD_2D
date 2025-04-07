%{
Derive adjoint form of MHD equations
%}

clear;

addpath("functions\");

n = 256;

nu  = 1/100;
eta = 1/100;

[x,y] = meshgrid( (0:(n-1))/n*2*pi );

seed = 0;
fields = random_initial_data( n, seed );

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



%integrate a transient
transient_steps = 4616; %1024*30;
vel_mag = zeros(transient_steps,1);

for i = 1:transient_steps
  i/transient_steps
  k1 = dt*state_vel( fields, d, params );
  k2 = dt*state_vel( fields + k1/2, d, params );
  k3 = dt*state_vel( fields + k2/2, d, params );
  k4 = dt*state_vel( fields + k3,   d, params );

  vel_mag(i) = norm(k1(:));

  fields = fields + ( k1 + 2* k2 + 2* k3 +  k4)/6;

  if mod(i,100) == 0
    figure(1);
    visualize_fields(fields);
    
    figure(2);
    semilogy( 1:i, vel_mag(1:i) );

    drawnow;
  end
end




%%
visualize_fields( fields);
save("late.mat", "d", "params", "fields", "dt", "transient_steps");

%% Let's use this late time state to 
clear;
load("late.mat");

steps = 256;
every = 40;
n = params.n;
traj = zeros(n,n,2,steps);

traj(:,:,:,1) = fields;
for i = 2:steps
  i/steps
  for j = 1:every
    k1 = dt*state_vel( fields, d, params );
    k2 = dt*state_vel( fields + k1/2, d, params );
    k3 = dt*state_vel( fields + k2/2, d, params );
    k4 = dt*state_vel( fields + k3,   d, params );
    
    fields = fields + ( k1 + 2* k2 + 2* k3 +  k4)/6;
  end
  traj(:,:,:,i) = fields;
end

%% Cross recurrence
traj2 = reshape( fft2(traj), [], steps);
traj2 = abs(traj2);

max_sep = 64;
dist = zeros(max_sep, steps);
for i = 1:steps
  i/steps
  idx = 1:min(max_sep, steps-i);
  future = traj2(:, i + idx);
  dist(idx,i) = vecnorm( future - traj2(:,i) );
end

dist = dist ./ mean(dist, "all");

%%
clf;
imagesc(dist);
colorbar();
clim([0,1]);
set(gca,"ydir", "normal");
pbaspect([steps, max_sep, 1]);
colormap jet
return
%% examine a candidate

delay = 15; %61; %15;%12;%11;
start = 499;%162;%794; %872;%421;

figure(1)
visualize_fields( traj(:,:,:,start) );

figure(2)
visualize_fields( traj(:,:,:,start + delay) );



%% Adjoint looping

fields = traj(:,:,:,start);
params.steps = delay*every;
T = delay*every*dt;
sx = 0.0;
z = [fields(:); T; sx];

%%
load("candidates/candidate_4.mat");
addpath("functions/");

z = gpuArray(z);
names = fieldnames(d);
for i = 1:numel(names)
  d.(names{i}) = gpuArray(d.(names{i}));
end

maxit = 32;

lr = 1e-3;
beta1 = 0.85;
beta2 = 0.999;
epsilon = 1e-6;

%How do you lower a tangent vector into a covector
metric = @(v) v; %Euclidean metric
%metric = @(v) real(ifft2( d.inv_lap .* fft2(v)));

loss_fn = @(z) wrapper(z, params, d, metric);
[z, fs] = adam_optimization( loss_fn, z, lr, maxit, beta1, beta2, epsilon);
%[z, fs] = adagrad( loss_fn, z, lr, maxit, epsilon);

save( "candidates/candidate_4.mat", "z", "params", "d" );


%%
%figure(3);
%semilogy(fs);
addpath("functions/");
load("candidates/candidate_4.mat");

T = z(end-1);
sx = z(end);
n = params.n;
fields_out = reshape(z(1:end-2), [n,n,2]);

figure(1)
visualize_fields(fields_out);
drawnow;


dt = T/params.steps;
for i = 1:params.steps
    k1 = dt*state_vel(fields_out, d, params);
    k2 = dt*state_vel(fields_out + k1/2, d, params);
    k3 = dt*state_vel(fields_out + k2/2, d, params);
    k4 = dt*state_vel(fields_out + k3, d, params);
    
    fields_out = fields_out + (k1 + 2*k2 + 2*k3 + k4)/6;

   if mod(i,8) ~= 0
      continue;
   end

    figure(2);
    visualize_fields(fields_out);
    drawnow;
end

visualize_fields(fields_out);













function [f, df] = wrapper(z, params, d, metric)
  T = z(end-1);
  sx= z(end);
  n = params.n;
  fields = reshape(z(1:end-2), [n,n,2]);

  [f, dfdx, dfdT, dfdsx ] = adjoint_looping( fields, T, sx,  params, d, metric );

  df = [dfdx(:); dfdT; dfdsx];
end