'''
Let's create a recurrence diagram from DNS and look for RPO candidates
'''

import jax
import jax.numpy as jnp
import time
#import matplotlib.pyplot as plt

import lib.mhd_jax as mhd_jax

from scipy.io import savemat

n  = 512 #spatial resolution
dt = 1/1024 #timestep
precision = jnp.float64

steps = 1024*16

nu  = 1/400
eta = 1/400
b0  = [0.0, 0.0] # Mean magnetic field


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
f = f.at[0, :, :].set( jnp.cos(x-0.1)*jnp.sin(x+y-1.2) - jnp.sin(3*x-1)*jnp.cos(y-1) + 2*jnp.cos(2*x-1))
f = f.at[1, :, :].set( jnp.cos(x+2.1)*jnp.sin(y+3.5) - jnp.cos(1-x) )
    
#fft the data before we evolve
f = jnp.fft.rfft2(f)

print("This script is testing the runtime performance of two fourth order integration schemes:")
print("The first is standard RK4, which requires four velocity evalutions per step.")
print("The second is Two Derivative RK4 (TDRK4), which requires one velocity evaluation and two acceleration evaluations.")
print("While TDRK4 might be cheaper in theory, it would require the programmer to write another function for the second time derivative.")
print("This is costly and error prone, so such methods have not seen wide adoption.")
print("This script demonstrates this is no longer the case.")
print("We can use automatic differentiation to get the second derivative \'for free\' at compile time.\n")

################################
#Integrate forward with TDRK4
################################
evolve = jax.jit( lambda f, steps: mhd_jax.tdrk4(f, dt, steps, param_dict) )
_ = evolve(f, 1) #JIT Compile with one step

start = time.time()
f_tdrk4 = evolve(f, steps)
stop = time.time()
print(f"Integrating {steps} steps with TDRK4 (using autodiff for Jacobian-vector products) took {stop-start} second...")
f_tdrk4 = jnp.fft.irfft2(f_tdrk4)




############################################################
#Integrate with standard RK4 to compare output and walltime
############################################################

evolve = jax.jit( lambda f, steps: mhd_jax.rk4(f, dt, steps, param_dict) )
_ = evolve(f, 1) #JIT Compile with one step

start = time.time()
f_rk4 = evolve(f, steps)
stop = time.time()
print(f"Integrating {steps} steps with RK4 took {stop-start} second...")
f_rk4 = jnp.fft.irfft2(f_rk4)


#Visualize both outputs in MATLAB
savemat( "tdrk4.mat", {"f_tdrk4": f_tdrk4, "f_rk4": f_rk4} )