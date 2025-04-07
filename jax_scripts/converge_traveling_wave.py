'''
Let's converge equilibria/traveling waves
'''
import jax
import jax.numpy as jnp

import mhd_jax_v2 as mhd_jax
import adam

from scipy.io import savemat, loadmat

# Simulation parameters
n = 128  # grid resolution
precision = jnp.float32  # Double or single precision

# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

# Generate grid information
param_dict = mhd_jax.construct_domain(n, precision)

nu  = 1/40  # hydro dissipation
eta = 1/40  # magnetic dissipation

x = param_dict['x']
y = param_dict['y']

# Mean magnetic field
b0 = [0.0, 0.1]

# Construct your forcing
forcing = -4*jnp.cos(4*y)

# Mean magnetic field
b0 = [0.0, 0.1]

# Construct your forcing
forcing = -4*jnp.cos(4*y)

param_dict.update({'forcing': forcing, 'b0': b0, 'nu': nu, 'eta': eta})







# Initial data
f = jnp.zeros([2, n, n], dtype=precision)

# EQ1
#f = f.at[0,:,:].set( jnp.cos(x) + jnp.sin(3*x-1)*jnp.cos(y-1) )

# EQ2
#f = f.at[0,:,:].set( jnp.cos(x) + jnp.cos(y) )

# EQ3
#f = f.at[1,:,:].set( jnp.cos(x) + jnp.cos(y) )


# EQ4
f = f.at[0,:,:].set( jnp.cos(x+2*y-1) + jnp.cos(x-y) )
f = f.at[1,:,:].set( jnp.cos(x) + jnp.cos(y + 1) )


# f = f.at[1,:,:].set( jnp.cos(x+2.1)*jnp.sin(y+3.5) )
# f = jnp.fft.rfft2(f)
# f = f.at[0,:,:].set( jnp.cos(x) + jnp.cos(y) )
# f = f.at[1,:,:].set( jnp.cos(x+2.1)*jnp.sin(y+3.5) )

maxit = 1024*16

wave_speed = 0.25
input_dict = {'fields': f, 'wave_speed': wave_speed}

loss_fn = jax.jit(mhd_jax.loss_traveling_wave)
grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

m, v = adam.init_adam(input_dict)
update_fn = jax.jit(adam.adam_update)

for t in range(maxit):
    loss = loss_fn( input_dict, param_dict )
    grad = grad_fn( input_dict, param_dict)

    # ADAM optimization
    input_dict, m, v = update_fn( input_dict, grad, m, v, t+1, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-6)

    '''
    # Mask high Fourier modes
    f = jnp.fft.rfft2(f)
    f = jnp.fft.irfft2(mask*f)
    '''
    if (t % 128 == 1):
        print(f"{t}: loss = {loss}")

savemat("solutions/eq/EQ4.mat", {"f": input_dict['fields'], "c": input_dict['wave_speed'] })
