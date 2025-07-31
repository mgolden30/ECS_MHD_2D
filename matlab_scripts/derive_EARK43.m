%{
Let's numerically derive an adaptive RK scheme with operator splitting.
%}


clear;
rng(3);

s = 5;
x = rand( 4*s + s*(s+1), 1 );

maxit = 1024;
h = 1e-3;

for i = 1:maxit
  f = objective(x,s);
  if mod(i,128) == 0
    fprintf("%d: |f| = %e\n", i, norm(f) );
  end

  J = zeros( numel(f), numel(x) );

  for j = 1:numel(x)
    x2 = x;
    x2(j) = x2(j) + h;

    J(:,j) = (objective(x2,s) - f)/h;
  end

  inner = 15;
  [dx, ~] = lsqr(J,f,1e-4,inner);
  x = x - 0.1*dx;
end

[a, q, b, b2, r, r2] = unpack(x,s);

%Print the modified Butcher table
[a,q]
[b.', r.']
[b2.', r2.']
return

%% Quick convergence test to check that my methods are the desired orders
[a, q, b, b2, r, r2] = unpack(x,s);



aa = 0.2;
bb = 0.2;
cc = 5.7;

%Linear part of Rossler dynamics
L = [
    0, -1, -1;
    1, aa,  0;
    0,  0,  -1;
    ];
%L = -0.1*eye(3);


%Just the nonlinear parts of the dynamics
v = @(x) [ 0; 0; bb + x(3)*(x(1) - cc)];

%initial condition
y0 = [1;2;3];

t = 10;
steps = [200, 400, 512, 600, 750, 870];

steps_fine = 2^14;

y_fine = ea_runge_kutta(y0, t, steps_fine, a, b, q, r, v, L);

err = [];
err2= [];

for i = 1:numel(steps)
  y = ea_runge_kutta(y0, t, steps(i), a, b, q, r, v, L);
  err(i) = norm(y - y_fine);

  y = ea_runge_kutta(y0, t, steps(i), a, b2, q, r2, v, L);
  err2(i) = norm(y - y_fine);
end

scatter( steps, err );
hold on
scatter( steps, err2 );
hold off

set(gca, "xscale", "log");
set(gca, "yscale", "log");

p  = polyfit( log(steps), log(err), 1 );
p2 = polyfit( log(steps), log(err2), 1 );

fprintf("Estimated error of method #1 is" + p(1)  + "\n");
fprintf("Estimated error of method #2 is" + p2(1) + "\n");




function x = ea_runge_kutta( x, t, steps, a, b, q, r, v, L )
  %{
  exponential ansatz Runge-Kutta
  %}

  h = t/steps;

  s = size(a,1);
  k = zeros(numel(x), s);

  c = sum(a,2);

  for i = 1:steps
    for j = 1:s
      xt = expm( h*c(j)*L )*x;
      for m = 1:j-1
        xt = xt + a(j,m)*expm( h*q(j,m)*L )*k(:,m);
      end
      k(:,j) = h*v(xt);
    end

    %Compute next step
    x = expm(h*L)*x;
    for m = 1:s
      x = x + b(m) * expm( h*r(m)*L ) * k(:,m);
    end
  end
end



function [a, q, b, b2, r, r2] = unpack(x,s)
  b  = x(0*s + (1:s));
  b2 = x(1*s + (1:s));
  r  = x(2*s + (1:s));
  r2 = x(3*s + (1:s));

  q = zeros(s,s);
  a = zeros(s,s);

  k = 4*s + 1;
  for i = 1:s
    for j = 1:i-1
      a(i,j) = x(k); k = k+1;
      q(i,j) = x(k); k = k+1;
    end
  end
end

function f = objective( x, s )
  [a, q, b, b2, r, r2] = unpack(x,s);
  
  c = sum(a,2);

  %This is a helpful shorthand
  aq = sum(a.*q,2);

  f = [
    %Order conditions
    sum(b) - 1;
    sum(b.*c) - 1/2;
    sum(b.*r) - 1/2;
    sum(b.*(a*c)) - 1/6;
    sum(b.*r.*c) - 1/6;
    sum(b.*r.^2) - 1/3;
    sum(b.*aq) - 1/6;
    sum(b.*c.^2) - 1/3; 
    
    %Fourth order conditions
    sum(b.*a*a*c) - 1/24;
    sum(b.*r.*a*c) - 1/24;
    sum(b.*(a.*q)*c) - 1/24;
    sum(b.*a*aq) - 1/24;
    sum(b.*r.*r.*c) - 1/12;
    sum(b.*r.*aq) - 1/24;
    sum(b.*sum(a.*q.*q,2)) - 1/12;
    sum(b.*r.^3) - 1/4;
    sum(b.*c.*a*c) - 1/8;
    sum(b.*c.*aq) - 1/8;
    sum(b.*a*c.^2) - 1/12;
    sum(b.*r.*c.*c) - 1/12;
    sum(b.*c.^3) - 1/4;

    %Order conditions for lower order method
    sum(b2) - 1;
    sum(b2.*c) - 1/2;
    sum(b2.*r2) - 1/2;
    sum(b2.*(a*c)) - 1/6;
    sum(b2.*r2.*c) - 1/6;
    sum(b2.*r2.^2) - 1/3;
    sum(b2.*aq) - 1/6;
    sum(b2.*c.^2) - 1/3; 

    
    %FSAL property
    a(end,:).' - b;
    q(end,:).' - r;

    
    %Conditions for "nice" coefficients
    a(3,1);
    a(4,1);
    a(4,2);
    a(2,1) - 0.5;
    a(3,2) - 0.5;
    a(4,3) - 1.0;

    q(3,1);
    q(4,1);
    q(4,2);

    r - r2;
    b2(4) - 1/3;
  ];
end