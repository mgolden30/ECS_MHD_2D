%{
This is a script to help me understand Lanczos iteration and how
tridiagonalization works
%}

n = 64;

rng(1);
A = randn(n,n);

%Symmetrize A
A = A + A.';

%Allocate memory
T = zeros(n,n);
V = zeros(n,n);

%Start with a random vector
v = randn(n,1);
v = v / norm(v);

%matrix multiplication
w = A*v;

%orthogonalize
a = my_dot(w, v);
w = w - a*v;

V(:,1) = v;
T(1,1) = a;

for j = 2:n
  b = norm(w);
  T(j-1,j) = b;
  T(j,j-1) = b;

  v = w/b;
  V(:,j) = v;

  %Evaluate
  w = A*v;
  a = my_dot(w,v);
  T(j,j) = a;
  w = w - a*v - b*V(:,j-1);

  %Orthogonalize w.r.t all previous basis vectors
  %for k = 1:j-2
  %  w = w - dot(w, V(:,k))*V(:,k);
  %end
end

imagesc( log10(abs(V*T - A*V)) );
colorbar();


function sum = KahanSum( v )
  %running sum
  sum = 0.0;

  %running compensation for lost bits
  c = 0.0;

  for i = 1:numel(v)
    y = v(i) - c;
    t = sum + y;
    c = (t-sum) - y;
    sum = t;
  end
end


function sum = my_dot( u, v )
  %sum = KahanSum( u.*v );
  sum = dot(u,v);
end
