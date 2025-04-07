function data = random_initial_data( n, seed )
  rng(seed);

  k = 0:n-1;
  k(k>n/2) = k(k>n/2) - n;
  mask = abs(k) < 4;
  mask(1) = 0;

  data = (2*rand(n,n,2) - 1) + 1i*(2*rand(n,n,2) - 1);
  data = data.*mask.*mask.';

  data = real(ifft2(data));

  data = 5 * data / max(abs(data), [], "all");
end