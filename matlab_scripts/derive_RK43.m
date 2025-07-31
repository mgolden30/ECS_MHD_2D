%{
Let's numerically derive an adaptive RK scheme.
%}


clear;
rng(1);

s = 5;
x = rand( 2*s + s*(s+1)/2,1 );

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
  [dx, ~] = lsqr(J,f,1e-3,inner);
  x = x - 0.1*dx;
end


[a, b, b2] = unpack(x,s)
%c = sum(a,2);
%sum(b2.*c.^2)/2 
%sum(b2.*(a*c))


%% Quick convergence test to check that my methods are the desired orders

aa = 0.2;
bb = 0.2;
cc = 5.7;
v = @(x) [-x(2)-x(3); x(1) + aa*x(2); bb + x(3)*(x(1) - cc)];


x0 = [1;2;3];


t = 5;
steps = [128, 256, 512, 1024];

steps_fine = 4096;

x_fine = runge_kutta(x0, t, steps_fine, a, b, v);


err = [];
err2= [];

for i = 1:numel(steps)
  x = runge_kutta(x0, t, steps(i), a, b, v);
  err(i) = norm(x - x_fine);

  x = runge_kutta(x0, t, steps(i), a, b2, v);
  err2(i) = norm(x - x_fine);
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


function x = runge_kutta( x, t, steps, a, b, v )
  h = t/steps;

  s = size(a,1);
  k = zeros(numel(x), s);

  for i = 1:steps
    k(:,1) = h*v(x);
    for j = 2:s
      k(:,j) = h*v(x + k * a(j,:).');
    end
    x = x + k*b;
  end
end






function [a, b, b2] = unpack(x,s)
  b  = x(1:s);
  b2 = x(s + (1:s));

  a = zeros(s,s);

  k = 2*s + 1;
  for i = 1:s
    for j = 1:i-1
      a(i,j) = x(k);
      k = k + 1;
    end
  end
end

function f = objective( x, s )
  [a, b, b2] = unpack(x, s);

  c = sum(a,2);

  f = [
    %Fourth order constraints
    sum(b) - 1;
    sum(b.*c) - 1/2;
    sum(b.*c.^2) - 1/3;
    sum(b.*(a*c)) - 1/6;
    sum(b.*c.^3) - 1/4;
    sum(b.*a*c.^2) - 1/12;
    sum(b.*c.*a*c) - 1/8;
    sum(b.*a*a*c) - 1/24;
    
    %Second order constraints
    sum(b2) - 1;
    sum(b2.*c) - 1/2;
    sum(b2.*c.^2) - 1/3;
    sum(b2.*(a*c)) - 1/6;

    %Other constraints
    %{
    a(3,1);
    a(4,1);
    a(4,2);
    a(2,1) - 1/2;
    a(3,2) - 1/2;
    a(4,3) - 1;
    %}
    
    %Pretty natural choice
    c - [0; 1/3; 2/3; 1; 1]

    %First Same As Last for step reuse
    a(5,1:4).' - b(1:4);

    %Enforce that the errors in the fourth order conditions are the same
    %sum(b2.*c.^3)/6 + sum(b2.*a*c.^2)/2;
    %sum(b2.*c.^3)/6 + sum(b2.*c.*a*c)/3;
    %sum(b2.*c.^3)/6 + sum(b2.*a*a*c);
    %b2(5) - 1/8;
    b2(4);
  ];
end