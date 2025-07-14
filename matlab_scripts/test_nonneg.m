clear;

n = 32;

rng(1);
A = randn(n,n);
b = randn(n,1);

epsilon = 1e-12;

x = lsqnonneg(A,b);
%x = nonpos_lsqr(A, b, epsilon);

plot(x)

function x = nonpos_lsqr(A, b, epsilon)
  m = size(A,1);
  n = size(A,2);

  P = zeros(n,1); %Passive set
  R = ones(n,1);  %Active set

  x = zeros(n,1);
  w = A.'*(b - A*x);
  
  while max(w(R)) > epsilon
    [~, j] = max(w(R));

    P(j) = 1;
    R(j) = 0;

    P = P==1;
    R = R==1;

    AP = A(:, P);
    
    s = zeros(n,1);
    s(P) = ( AP.'* AP )\ (AP.'*b);
    
    while min(s(P)) <= 0
      diff = x./(x-s);
      keep = P & (s < 0);

      alpha = min( diff(keep) );
      x = x + alpha *(s-x);

      move_to_R = P & (x<=0);

      P(move_to_R) = 0;
      R(move_to_R) = 1;
      
      P = P==1;
      R = R==1;
      AP = A(:, P);

      %sum(P)
      %size(AP)
      %numel( ( AP.'* AP )\ (AP.'*b) )

      s(P) = ( AP.'* AP )\ (AP.'*b);
      s(R) = 0;
    end
    x = s;
    w = A.'*(b - A*x);
  end
end