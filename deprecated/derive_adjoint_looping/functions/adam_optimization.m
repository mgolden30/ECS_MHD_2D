
function [params, fs] = adam_optimization(loss_fn, params, learning_rate, num_iters, beta1, beta2, epsilon)
    % Adam optimization algorithm
    % 
    % Inputs:
    % - loss_fn: function handle, returns [loss, grad]
    % - params: initial parameters (column vector)
    % - learning_rate: scalar, learning rate
    % - num_iters: number of iterations
    % - beta1: scalar, exponential decay rate for first moment estimates
    % - beta2: scalar, exponential decay rate for second moment estimates
    % - epsilon: small scalar to prevent division by zero
    %
    % Outputs:
    % - params: optimized parameters
    
    % Initialize moment estimates
    m = zeros(size(params)); % first moment vector
    v = zeros(size(params)); % second moment vector
    t = 0; % time step

    fs = zeros(num_iters,1);

    % Optimization loop
    for iter = 1:num_iters
        % Increment time step
        t = t + 1;
        
        % Compute loss and gradient
        tic
        [loss, grad] = loss_fn(params);
        walltime = toc;
        fs(iter) = loss;

        % Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * grad;
        v = beta2 * v + (1 - beta2) * (grad.^2);
        
        % Compute bias-corrected moment estimates
        m_hat = m / (1 - beta1^t);
        v_hat = v / (1 - beta2^t);
        
        % Update parameters
        params = params - learning_rate * m_hat ./ (sqrt(v_hat) + epsilon);
        
        % Optional: Display progress
        fprintf('Iteration %d, Loss: %.6f, Walltime: %.3f, T: %.3f, sx: %.3f\n', iter, loss, walltime, params(end-1), params(end) );
    end
end