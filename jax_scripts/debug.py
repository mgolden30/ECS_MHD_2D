'''
Let's debug a function for jax to do adjoint descent without blowing up our memory requirements
'''

import time
import jax
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
import lib.adam as adam

from scipy.io import savemat, loadmat

###############################
# Construct numerical grid
###############################

n = 128 # grid resolution
precision = jnp.float64  # Double or single precision

# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

# Generate grid information
param_dict = mhd_jax.construct_domain(n, precision)

#Pull out grid matrices for forcing construction
x = param_dict['x']
y = param_dict['y']




#################################################
# Physical parameters: dissipation, forcing, ...
#################################################
nu  = 1/40  # hydro dissipation
eta = 1/40  # magnetic dissipation

# Mean magnetic field
b0 = [0.1, 0.0]

# Construct your forcing
forcing = -4*jnp.cos(4*y)

param_dict.update({'forcing': forcing, 'b0': b0, 'nu': nu, 'eta': eta})


#Load the turbulent trajectory
data = jnp.load("turb.npz")
fs = data['fs']

#MATLAB indices I picked from visually inspecting the recurrence diagram.
idx = [181, 205]

f = fs[idx[0], :, :, :]
f = jnp.fft.irfft2(f)

dt = 0.01
ministeps = 32
T = dt * ministeps * (idx[1] - idx[0])
sx = 0.0


###########################################
# Load an initial condition from turbulence
###########################################




#Create a dictionary of optimizable field
input_dict = {"fields": f, "T": T, "sx": sx}

#load a previous guess
matlab_data = loadmat("data/RPO_candidate_4152.mat")
input_dict = {"fields": matlab_data['fields'], "T": matlab_data['T'][0][0], "sx": matlab_data['sx'][0][0] }


#Add the number of steps we need
param_dict.update({ 'steps': ministeps * (idx[1] - idx[0])*10 } )



###############################
# Adjoint descent time
###############################

loss_fn = lambda input_dict: loss_functions.loss_RPO(input_dict, param_dict)

#Define a function to compute the vlaue of the loss and the gradient simultaneously
grad_fn = jax.jit(jax.value_and_grad(loss_fn))

'''
#Compile
_ = grad_fn(input_dict)

#Try OG loss function
start = time.time()
loss, grad = grad_fn(input_dict)
stop = time.time()
walltime = stop-start

print(f"walltime = {walltime}, loss = {loss}, grad_T = {grad['T']}, |grad_f| = {jnp.mean( jnp.square( grad['fields']))}")
'''

#Determine the number of segments
segments = 8
#grad_fn  = jax.jit(lambda input_dict: loss_functions.loss_RPO_memory_efficient( input_dict, param_dict, segments ))
grad_fn  = lambda input_dict: loss_functions.loss_RPO_memory_efficient( input_dict, param_dict, segments )
grad_fn = jax.jit(grad_fn)

#Compile
_ = grad_fn(input_dict)

print(param_dict['steps'])

start = time.time()
loss, grad = grad_fn(input_dict)
stop = time.time()
walltime = stop-start

print(f"walltime = {walltime}, loss = {loss}, grad_T = {grad['T']}, |grad_f| = {jnp.mean( jnp.square( grad['fields']))}")
