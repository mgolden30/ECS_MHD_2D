%{
I am interested in nonlinear extensions of Newton-Raphson iteration. 

TENSOR METHODS FOR NONLINEAR EQUATIONS*
ROBERT B. SCHNABEL? AND PAUL D. FRANK?
%}



%Define a function I want to compute the root of
n = 128; %dimension
rng(1); %seed random number generation 
A = randn(n,n);
b = randn(n,1);

%Make A kinda sparse
A(randn(size(A)) < 1) = 0;

%Nonlinear function I want to do root finding for.
F = @(x) cos( A*x - b );

%Start with an initial guess of a root 
x = randn(n,1);

h = 1e-3; %finite difference parameter
inner = 64; %Krylov subspace dimension
p = 3;

tensor_GMRES_test( x, F, h, inner, p )

function [Q, H] = Arnoldi( x, F, h, inner )
  %Just finite difference the Jacobian
  n = numel(x);
  
  Q = zeros(n, inner+1);
  H = zeros(inner+1, inner);

  F0 = F(x);
  Q(:,1) = F0 / norm(F0);

  for k = 1:inner
      % Compute the finite difference for the Jacobian
      xPlus = x + h * Q(:,k);
      FPlus = F(xPlus);
      Jq = (FPlus - F0) / h; % Jacobian approximation
      for i = 1:k
        H(i,k) = dot( Jq, Q(:,i) ); 
        Jq = Jq - H(i,k)*Q(:,i);
      end
      H(k+1,k) = norm(Jq);
      Q(:,k+1) = Jq / H(k+1,k);
  end
end


function tensor_GMRES_test( x, F, h, inner, p )
  %{
  p - number of quadratic directions
  %}
  n = numel(x);
  f = F(x);

  [Q, H] = Arnoldi( x, F, h, inner );
  
  [U, S, V] = svd(H);

  size(H)
  size(S)
  size(V)

  %Compute the second derivative along the small singular vectors
  A = zeros(n,p);

  size(x)
  size(V(:,end-p+1))

  for i = 1:p
    v = Q(:,1:end-1) * V(:,end-p+i); %perturbation vector in physical space.
    xm = x - h*v;
    xp = x + h*v;
    A(:,i) = (F(xm) - 2*f + F(xp))/h/h;
  end

  %Use the second derivative information to enlarge the Krylov subspace
  for i = 1:p
    a = A(:,i);
    a = a - Q * (Q.' * a);
    Q(:,end+1) = a / norm(a); %dynamic memory. Change eventually.
  end

  %Rotate the Hessenberg form of the Jacobian to act on right singular
  %vectors.
  H_right = H * V;
  
  %Extend H with p rows of zeros
  H_right = [H_right; zeros(p, size(H,2))];

  %Make H_right upper triangular.
  [W, H_] = qr(H_right);

  %Transform A and f into the Krylov basis and then upper triangular basis.
  F_ = W.'*Q.'*f;
  A_ = W.'*Q.'*A;

  size(A_)
  size(H_)

  %Solve the nonlinear subproblem with Newton-Raphson
  

end

function x = solve_subproblem( a, B, C, p )
  %Minimize \| a + Bx + C x.^2  \|_2
  
  r = @(x) a + B*x + C*(x.^2);
  
  x = zeros(p,1);
  
  maxit = 128;
  for i = 1:maxit
    J = 
  end
end