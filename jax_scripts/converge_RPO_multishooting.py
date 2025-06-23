'''
Let's use JAX to hunt for RPOs. I got it working for single shooting, but the memory requirement is quite nasty.
I think multishooting is the solution. Break up integration into segements. Compute grad one segment at a time.
'''

import lib.mhd_jax as mhd_jax
import time
import jax
import jax.numpy as jnp

from scipy.io import savemat, loadmat

# Simulation parameters
n = 256  # grid resolution
precision = jnp.float64  # Double or single precision

# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

# Generate grid information
x, y, kx, ky, mask, to_u, to_v = mhd_jax.construct_domain(n, precision)

nu = 1/100  # hydro dissipation
eta = 1/100  # magnetic dissipation

# Mean magnetic field
b0 = [0.0, 1.0]

# Construct your forcing
forcing = -4*jnp.cos(4*y)

# Initial data
segments = 8 # Break up the trajectory into this many segments
f = jnp.zeros([segments, 2, n, n], dtype=precision)

# Load a single shooting guess and use it to construct the multishooting initial state
#my_dict = loadmat("converged_RPO_14721.mat")

T = my_dict["T"][0][0]
sx = my_dict["sx"][0][0]
f_mini = my_dict["f"]
f_mini = jnp.fft.rfft2(f_mini)

#I hard-coded 256 timesteps into single shooting due to memory constraints
h = T/256 

mini_steps = round(256 / segments)

jit_rk4_step = jax.jit(mhd_jax.rk4_step)

for i in range(segments):
    f_mini = jnp.fft.irfft2(f_mini)
    f = f.at[i, :, :, :].set(f_mini)
    
    f_mini = jnp.fft.rfft2(f_mini)
    for j in range(mini_steps):
        f_mini = jit_rk4_step( f_mini, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, h)


# FOr multishooting, I want T and sx to be period and shift per segment
T = T/segments
sx= sx/segments



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
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads ** 2)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Update parameters
    params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    return params, m, v

steps = 1024

z = mhd_jax.pack_RPO_multi( f, T, sx, n, segments)

m = 0*z
v = 0*z

maxit = 1000000

loss_fn = jax.jit(mhd_jax.loss_RPO_multi)
grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

update_fn = jax.jit(adam_update)

my_dict = loadmat("data/converged_RPO_4225.mat")

T = my_dict["T"][0][0]
sx = my_dict["sx"][0][0]
f= my_dict["f"]

z = mhd_jax.pack_RPO_multi(f, T, sx, n, segments)


for t in range(maxit):

    start = time.time()
    loss = 0.0
    for seg in range(segments):
        loss = loss + loss_fn(z, steps, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, n, seg)
    loss_time = time.time() - start

    start = time.time()
    grad = 0*z
    for seg in range(segments):
        grad = grad + grad_fn(z, steps, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, n, seg)
    grad_time = time.time() - start

    # ADAM optimization
    #lr = 1e-2 got pretty close  p
    z, m, v = update_fn(z, grad, m, v, t+1, lr=5e-4, beta1=0.9, beta2=0.999, eps=1e-6)

    f, T, vx = mhd_jax.unpack_RPO_multi(z)
    
    f = jnp.fft.rfft2(f)
    f = f * mask
    f = jnp.fft.irfft2(f)
    
    z =  mhd_jax.pack_RPO_multi(f, T , sx, n, segments)

    #if (t % 128 == 1):
    print(f"{t}: loss = {loss}, loss_time = {loss_time}, grad_time = {grad_time}")

    if ( t % 128 == 1 ):
        f, T, sx = mhd_jax.unpack_RPO_multi(z)
        savemat( f"data/converged_RPO_{t}.mat", {"f": f, "T": T, "sx": sx})