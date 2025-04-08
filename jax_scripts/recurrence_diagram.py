'''
Let's create a recurrence diagram from DNS and look for RPO candidates
'''

import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

import lib.mhd_jax as mhd_jax

from scipy.io import savemat


n  = 128
dt = 0.0025
#transient_steps = 8*1024
#steps = 1024
ministeps = 64
precision = jnp.float64

transient_steps = 16*1024
steps = 512

nu  = 1/100
eta = 1/100
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

# Initial data
f = jnp.zeros([2, n, n], dtype=precision)
f = f.at[0, :, :].set( jnp.cos(x-0.1)*jnp.sin(x+y-1.2) + jnp.sin(3*x-1)*jnp.cos(y-1) + 2*jnp.cos(x))
f = f.at[1, :, :].set( jnp.cos(x+2.1)*jnp.sin(y+3.5) )
    
#fft the data before we evolve
f = jnp.fft.rfft2(f)

############################
#Integrate a transient
############################
one_step = jax.jit( lambda f: mhd_jax.eark4(f, dt, 1, param_dict) )

start = time.time()
for i in range(transient_steps):
    f = one_step(f)
stop = time.time()
print(f"Transient of {transient_steps} steps took {stop-start} second...")

figure, axis = mhd_jax.vis(f)
figure.savefig("figures/post_transient.png", dpi=1000)
plt.close()


############################
# Integrate some turbulence
############################

one_step = jax.jit( lambda f: mhd_jax.eark4(f, dt, ministeps, param_dict) )


fs = jnp.zeros([steps,2,n,n//2+1], dtype=f.dtype )
start = time.time()
for i in range(steps):
    f = one_step(f)
    fs = fs.at[i,:,:,:].set(f)
stop = time.time()
print(f"Generating {steps} steps of turbulence took {stop-start} seconds.")


figure, axis = mhd_jax.vis(f)
figure.savefig("figures/post_evolution.png", dpi=1000)
plt.close()




##################################
# Construct a recurrence diagram
##################################

start = time.time()
dist = jnp.zeros([steps, steps])
for i in range(steps):
    #diff = fs - fs[i,:,:,:]    
    #diff = jnp.fft.irfft2(diff)

    diff = jnp.abs(fs) - jnp.abs(fs[i,:,:,:])    

    diff = jnp.reshape( diff, [steps, -1] ) #shape [steps, 2*n*n]

    dist = dist.at[:,i].set( jnp.linalg.vector_norm(diff, axis=1) )
stop = time.time()
print(f"Recurrence diagram computed in {stop - start} seconds")



import matplotlib.pyplot as plt

im = plt.imshow(dist/n/n, origin='lower', cmap='Pastel1')
plt.colorbar( im )
plt.title("recurrence")
plt.savefig("figures/recurrence.png", dpi=1000)

jnp.savez( "turb.npz", fs=fs, dt=dt, ministeps=ministeps )

fs = jnp.fft.irfft2(fs)
savemat( "dist.mat", {"dist": dist, "fs": fs} )