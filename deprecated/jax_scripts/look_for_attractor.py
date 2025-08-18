'''
Let's see if we can visualize an attractor
'''

import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO

from scipy.io import savemat

n  = 128
dt = 1/256

#visualize every
ministeps = 1

#number of states
samples = 128

#For visualization, float32 is fine
precision = jnp.float32

steps = round(100/dt)
print(steps)

nu  = 1/40
eta = 1/40
b0  = [0.0, 0.1] # Mean magnetic field

# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

# Construct a dictionary for grid information
param_dict = mhd_jax.construct_domain(n, precision)

# Get grids
x = param_dict['x']
y = param_dict['y']

forcing = -4*jnp.cos(4*y)

# Append the extra system information to param_dict
param_dict.update( {'nu': nu, 'eta': eta, 'b0': b0, 'forcing': forcing} )

#Append timestepping information as well
param_dict.update( {'dt': dt, 'ministeps': ministeps} )

# Initial data
#f = jnp.zeros([samples, 2, n, n], dtype=precision)
#f = f.at[0, :, :].set( jnp.cos(4*x-0.1)*jnp.sin(x+y-1.2) - jnp.sin(3*x-1)*jnp.cos(y-1) + 2*jnp.cos(2*x-1))
#f = f.at[1, :, :].set( jnp.cos(3*x+2.1)*jnp.sin(y+3.5) - jnp.cos(1-x) + jnp.sin(x + 5*y - 1 ) )

key = jax.random.PRNGKey(seed=0)
f = jax.random.normal( key, shape=[samples,2,n,n] )
f = f * jax.random.normal(key, shape=[samples,2,1,1] )

#fft the data before we evolve
f = jnp.fft.rfft2(f)
f = param_dict['mask'] * f

#Allocate memory
observables = jnp.zeros([steps, samples, 2])



############################
# Integrate some turbulence
############################

one_step = jax.jit( lambda f: mhd_jax.eark4(f, dt, ministeps, param_dict) )

#fs = jnp.zeros([steps,2,n,n//2+1], dtype=f.dtype )
start = time.time()

#I am learning that jax.lax.scan is a better way to gather forward time integration

def wrapper(f, _):
    f_new = one_step(f)
    return f_new, f_new #return carry and what to store

for i in range(steps):
    observables = observables.at[i,:,:].set( jnp.mean( jnp.square(jnp.fft.irfft2(f)), axis=[-1,-2] ) )
    f = one_step(f)
    print(i)

savemat("attractor.mat", {"observables": observables})
exit()

