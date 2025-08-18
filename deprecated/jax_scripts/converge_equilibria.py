'''
Let's use JAX to hunt for equilibria
'''

import mhd_jax

import jax
import time
import jax.numpy as jnp

from scipy.io import savemat, loadmat

######################################    
#Simulation parameters
#####################################

n  = 256   #grid resolution
precision = jnp.float64 #Double or single precision

#If you want double precision, change JAX defaults
if(precision == jnp.float64 ):
    jax.config.update("jax_enable_x64", True)

#Generate grid information
x, y, kx, ky, mask, to_u, to_v = mhd_jax.construct_domain(n, data_type=precision )

nu = 1/100 #hydro dissipation
eta= 1/100 #magnetic dissipation

#Mean magnetic field
b0 = [0.0, 0.1]

#Construct your forcing
forcing = -4*jnp.cos(4*y)


#Initial data
f = jnp.zeros( [2,n,n], dtype=precision )
#f = f.at[0,:,:].set( jnp.cos(x) + jnp.sin(3*x-1)*jnp.cos(y-1) )
#f = f.at[1,:,:].set( jnp.cos(x+2.1)*jnp.sin(y+3.5) ) 
#f = jnp.fft.rfft2(f)
f = f.at[0,:,:].set( jnp.cos(x) + jnp.cos(y) )
#f = f.at[1,:,:].set( jnp.cos(x+2.1)*jnp.sin(y+3.5) ) 







def adam_update(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
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
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads ** 2)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Update parameters
    params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    return params, m, v



my_dict = loadmat("converged.mat")
#my_dict = loadmat("jax_rk4.mat")
f = f.at[:,:,:].set( my_dict["f"] )

m = 0*f
v = 0*f


maxit = 1024*32


loss_fn = jax.jit( mhd_jax.equilibrium_loss )
grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

update_fn = jax.jit( adam_update )

for iteration in range(maxit):
    loss = loss_fn(f, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0)
    grad = grad_fn(f, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0)
    
    #ADAM optimization
    f, m, v = update_fn( f, grad, m, v, iteration+1, lr=1e-4, beta1=0.9, beta2=0.999, eps=1)

    #Mask high Fourier modes
    f = jnp.fft.rfft2(f)
    f = jnp.fft.irfft2(mask*f)

    if( iteration % 128 == 1):
        print(f"{iteration-1}: loss = {loss}")

savemat("converged.mat", {"f": f})