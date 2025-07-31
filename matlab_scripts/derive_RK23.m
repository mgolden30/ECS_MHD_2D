%{
Let's numerically derive an adaptive RK scheme.
%}


clear;
rng(13);

s = 4;
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

  [dx, ~] = lsqr(J,f,1e-3);
  x = x - 0.1*dx;
end

%%
[a, b, b2] = unpack(x,s)
c = sum(a,2);
sum(b2.*c.^2)/2 
sum(b2.*(a*c))

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
    %Third order constraints
    sum(b) - 1;
    sum(b.*c) - 1/2;
    sum(b.*c.^2) - 1/3;
    sum(b.*(a*c)) - 1/6;
    
    %Second order constraints
    sum(b2) - 1;
    sum(b2.*c) - 1/2;

    %Other constraints
    a(2,1) - 0.5;
    b(1) - 2/9;
    b(4) - 0;

    %First Same As Last for step reuse
    a(4,1:3).' - b(1:3);
    
    %Make the error in the third order points equal in magnitude
    sum(b2.*c.^2)/2 - sum(b2.*(a*c));
    b2(1) - 3/10;
  ];
end