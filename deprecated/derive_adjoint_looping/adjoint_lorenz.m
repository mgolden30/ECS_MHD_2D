%{
I want to refresh myself on adjoint looping. Adjoint looping is simply
gradient descent of a scalar loss function. In this case, take 

f(x,T) = 1/2 |x(0) - x(T)|^2

We need to compute the gradient with respect to both x and T. This script
covers.
%}


%Generate a numerical trajectory

x0 = [1;2;3];
h = 0.005;

%integrate a transient
for i = 1:4000
  x0 = rk4(x0, h, 1);
end


n = 1024;
traj = zeros(3, n);
traj(:,1) = x0;
h = 0.01;

for i = 2:n
  traj(:,i) = rk4( traj(:,i-1), h, 1 );
end

clf
scatter3( traj(1,:), traj(2,:), traj(3,:) );

%% Make a recurrence diagram

dist = zeros( n,n );

for i = 1:n
  dist(:,i) = vecnorm(traj - traj(:,i));
end

clf;
imagesc(dist)
colorbar();
axis square;
clim([0,1]);


%% Pick a PO candidate
idx = [500, 656];

i = idx(1):idx(2);

scatter3( traj(1,i), traj(2,i), traj(3,i) );

x = traj(:,idx(1));
T = (idx(2) - idx(1)) * h;
steps = idx(2) - idx(1);


%Evaluate the adjoint looping answer
[f, dfdT, dfdx] = adjoint_looping( x, T, steps );

%Define the objective
objective = @(x, T, steps) sqrt(sum( (x - rk4(x, T, steps)).^2 ));

%Check derivatives with finite differencing
dT = 1e-6;
dfdT_fd = (objective(x, T+dT, steps) - objective(x, T-dT, steps))/(2*dT);

diff_dT = norm(dfdT - dfdT_fd)

dx = 1e-6;
dfdx_fd = zeros(3,1);
for i = 1:3
  xp = x;
  xp(i) = xp(i) + dx;

  xm = x;
  xm(i) = xm(i) - dx;
  
  dfdx_fd(i) = (objective(xp, T, steps) - objective(xm, T, steps))/(2*dx);
end

diff_dx = norm( dfdx - dfdx_fd )



%% Attempt adjoint descent


x = traj(:,idx(1));
T = (idx(2) - idx(1)) * h;
steps = idx(2) - idx(1);

maxit = 1024;
lr = 1e-4;

fs = zeros(maxit,1);

for it = 1:maxit
  [fs(it), dfdT, dfdx] = adjoint_looping( x, T, steps );
 
  T = T - dfdT*lr;
  x = x - dfdx*lr;
end

semilogy( fs );


%% Attempt RK4 gradient descent

x = traj(:,idx(1));
T = (idx(2) - idx(1)) * h;
steps = idx(2) - idx(1);

z = [x;T];

maxit = 1024;
fs = zeros(maxit,1);

lr = 1e-3;
%{
for i = 1:maxit
  [fs(i), df] = nice(z,steps);

  %aim to
  z = z - lr*df;
end
plot(fs);
%}

beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-6;

loss_fn = @(z) nice(z, steps);
[z, fs] = adam_optimization( loss_fn, z, lr, maxit, beta1, beta2, epsilon);

semilogy(fs);




function [f, df] = nice(z, steps)
  [f, dfdT, dfdx] = adjoint_looping( z(1:3), z(4), steps );
  df = [dfdx; dfdT];
end

function A = linear_op()
  sigma = 10;
  rho = 28;
  beta = 8/3;

  A = [-sigma, sigma,0 ;
      rho, -1, 0;
      0,0,-beta];
end

function v = nonlinear_vel(x)
  %Just the nonlinear bits
  A = linear_op();
  v = [ 0; -x(1)*x(3); x(1)*x(2); ] + A*x;
end


function v = nonlinear_jac(x, y)
  %Just the nonlinear bits
  A = linear_op();
  v = [ 0; 
       -y(1)*x(3) - x(1)*y(3); 
       +y(1)*x(2) + x(1)*y(2); ] + A*y;
end

function v = nonlinear_jacT(x, y)
  %Trnaspose of Jacobian
  A = linear_op();
  %Just the nonlinear bits
  v = [ -x(3)*y(2) + x(2)*y(3);
         x(1)*y(3);
        -x(1)*y(2)] + A.' * y;
end


function x = rk4( x, T, steps )
  %{
  A simple second order scheme
  %}

  h = T/steps;

  for i = 1:steps
    k1 = h*nonlinear_vel( x );
    k2 = h*nonlinear_vel( x + k1/2 );
    k3 = h*nonlinear_vel( x + k2/2 );
    k4 = h*nonlinear_vel( x + k3 );

    x = x+ (k1 + 2*k2 + 2*k3 + k4)/6;
  end
end


function [f, dfdT, dfdx] = adjoint_looping( x, T, steps )
  %{
  PURPOSE:
  Compute the error in being a periodic orbit.

  f(x,T) =  sqrt( sum( x(0) - x(T) )^2 )

  In addition to this loss function, compute the *exact* 
  gradient with respect to period and initial position x.
  %}

  %For evaluating the transpose of the Jacobian, we will need the full
  %state space history.
  traj = zeros( 3, steps+1 );
  traj(:,1) = x;

  h = T/steps;


  %Let y denote gradient w respect to T
  y = 0*x;

  for i = 1:steps
    k1 =        h*nonlinear_vel(x);
    y1 = k1/T + h*nonlinear_jac(x, y);
  
    k2 =        h*nonlinear_vel(x + k1/2);
    y2 = k2/T + h*nonlinear_jac(x + k1/2, y + y1/2);

    k3 =        h*nonlinear_vel(x + k2/2);
    y3 = k3/T + h*nonlinear_jac(x + k2/2, y + y2/2);
  
    k4 = h*nonlinear_vel(x + k3);
    y4 = k4/T + h*nonlinear_jac(x + k3, y + y3);
  
    x = x + (k1 + 2*k2 + 2*k3 + k4)/6;
    y = y + (y1 + 2*y2 + 2*y3 + y4)/6;

    traj(:, i+1) = x;
  end

  %Compute difference vector
  u0 = traj(:,end) - traj(:,1);
  
  %Compute objective function
  f = norm(u0);

  %Gradient w.r.t period
  dfdT = dot( u0, y ) / f; 

  %RK4 constants
  b1 = 1/6;
  b2 = 1/3;
  b3 = 1/3;
  b4 = 1/6;

  a21 = 1/2;
  a32 = 1/2;
  a43 = 1;

  %update u
  u = u0;
  for i = steps:-1:1
    x = traj(:,i);

    k1 = h*nonlinear_vel(x);
    k2 = h*nonlinear_vel(x + k1*a21);
    k3 = h*nonlinear_vel(x + k2*a32);
    %k4 = h*nonlinear_vel(x + k3*a43);

    u4 = h*nonlinear_jacT(x + k3*a43, b4*u);
    u3 = h*nonlinear_jacT(x + k2*a32, b3*u + u4*a43);
    u2 = h*nonlinear_jacT(x + k1*a21, b2*u + u3*a32);
    u1 = h*nonlinear_jacT(x,          b1*u + u2*a21);
  
    u = u + u1 + u2 + u3 + u4;
  end

  dfdx = (u - u0)/f;
end










