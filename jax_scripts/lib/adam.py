'''
Define functions for easy use of ADAM optimization here
'''

import jax.numpy as jnp

def init_adam(params):
    '''
    Check if params is a dict
    '''

    if isinstance(params, dict):
        # If we have dictionaries, make a copy and set all keys to zero
        m = { key: 0*params[key] for key in params }
        v = { key: 0*params[key] for key in params }
    else:
        #Must be a single tensor
        m = 0*params
        v = 0*params
    return m, v


def adam_update(params, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Performs one update step of the ADAM optimizer for a single parameter array.

    :param params: NumPy array of parameters to be updated.
    :param grads: NumPy array of gradients (same shape as params).
    :param m: First moment estimate (same shape as params).
    :param v: Second moment estimate (same shape as params).
    :param t: Time step (integer, should be incremented after each update).
    :param lr: Learning rate.
    :param beta1: Decay rate for first moment estimate.
    :param beta2: Decay rate for second moment estimate.
    :param eps: Small number to prevent division by zero.
    :return: Updated parameters, updated first moment (m), updated second moment (v).
    """
    if isinstance(params, dict):
        m = {key: beta1 * m[key] + (1 - beta1) * grads[key] for key in params}
        v = {key: beta2 * v[key] + (1 - beta2) * (grads[key] ** 2) for key in params}

        # Bias correction
        m_hat = {key: m[key] / (1 - beta1 ** t) for key in params}
        v_hat = {key: v[key] / (1 - beta2 ** t) for key in params}

        # Update parameters
        params = {key: params[key] - lr * m_hat[key] / (jnp.sqrt(v_hat[key]) + eps) for key in params}


    else:
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update parameters
        params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    return params, m, v

