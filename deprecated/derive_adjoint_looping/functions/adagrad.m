function [parameters, losses] = adagrad(gradient_func, parameters, learning_rate, max_iters, epsilon)
    % AdaGrad Optimization Algorithm
    % 
    % Inputs:
    % gradient_func - Function handle that computes gradients and loss. 
    %                 Should be of the form [grad, loss] = gradient_func(parameters).
    % parameters - Initial parameter vector (column vector).
    % learning_rate - Initial learning rate (scalar).
    % max_iters - Maximum number of iterations.
    % epsilon - Small constant to avoid division by zero.
    %
    % Outputs:
    % parameters - Optimized parameters after training.
    % losses - Array of loss values at each iteration.
    
    % Initialize accumulated gradient squared sum
    grad_squared_sum = zeros(size(parameters));
    losses = zeros(max_iters, 1);
    
    for iter = 1:max_iters
        % Compute gradient and loss
        tic
        [loss, grad] = gradient_func(parameters);
        walltime = toc;

        losses(iter) = loss;
        
        % Update accumulated gradient squared sum
        grad_squared_sum = grad_squared_sum + grad.^2;
        
        % Compute adjusted learning rate
        adjusted_lr = learning_rate ./ (sqrt(grad_squared_sum) + epsilon);
        
        % Update parameters
        parameters = parameters - adjusted_lr .* grad;
        
        % Display progress
        fprintf('Iteration %d, Loss: %.6f, Walltime: %.3f\n', iter, loss, walltime);
    end
end
