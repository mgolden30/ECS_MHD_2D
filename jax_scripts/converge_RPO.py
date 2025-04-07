'''
Let's use JAX to hunt for RPOs with gradient descent.
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
b0 = [0.0, 0.1]

# Construct your forcing
forcing = -4*jnp.cos(4*y)

param_dict.update({'forcing': forcing, 'b0': b0, 'nu': nu, 'eta': eta})


#Load the turbulent trajectory
data = jnp.load("turb.npz")
fs = data['fs']

#MATLAB indices I picked from visually inspecting the recurrence diagram.
idx = [218, 268]
f = fs[idx[0], :, :, :]
f = jnp.fft.irfft2(f)

dt = 0.005
ministeps = 32
T = dt * ministeps * (idx[1] - idx[0])
sx = 0.0


###########################################
# Load an initial condition from turbulence
###########################################




#Create a dictionary of optimizable field
input_dict = {"fields": f, "T": T, "sx": sx}

#load a previous guess
#matlab_data = loadmat("data/RPO_candidate_4152.mat")
#matlab_data = loadmat("data/RPO_candidate_10000.mat")
#input_dict = {"fields": matlab_data['fields'], "T": matlab_data['T'][0][0], "sx": matlab_data['sx'][0][0] }


#Add the number of steps we need
param_dict.update({ 'steps': ministeps* (idx[1] - idx[0]) } )

print(f"using {param_dict['steps']} steps")


###############################
# Adjoint descent time
###############################

m, v = adam.init_adam(input_dict)
maxit = 10000000

#Define a function to compute the vlaue of the loss and the gradient simultaneously
loss_fn = lambda input_dict: loss_functions.loss_RPO(input_dict, param_dict)
grad_fn = jax.jit(jax.value_and_grad(loss_fn))


#Or do it memory efficient so we can go to many timesteps
segments = 8
grad_fn  = jax.jit(lambda input_dict: loss_functions.loss_RPO_memory_efficient( input_dict, param_dict, segments ))

#Compile
_ = grad_fn(input_dict)

#Jit the update routine for ADAM to attempt some speedup
update_fn = jax.jit(adam.adam_update)

for t in range(maxit):
    start = time.time()
    loss, grad = grad_fn(input_dict)
    stop = time.time()
    walltime = stop-start

    lr = 1e-2
    input_dict, m, v = update_fn(input_dict, grad, m, v, t+1, lr=lr, beta1=0.9, beta2=0.999, eps=1e-6)

    #dealias
    f = input_dict['fields']
    f = jnp.fft.rfft2(f) * param_dict['mask']
    f = jnp.fft.irfft2(f)
    input_dict['fields'] = f

    print(f"{t}: loss = {loss}, walltime: {walltime}, T = {input_dict['T']}")
    
    if ( t % 8 == 0 ):    
        macrosteps = param_dict['steps'] // ministeps
        
        f = input_dict["fields"]
        T = input_dict['T']
        dt= T/param_dict['steps']

        update = jax.jit( lambda f: mhd_jax.eark4(f, dt, ministeps, param_dict) )

        f = jnp.fft.rfft2(f)
        savemat( f"timeseries/0.mat", {"f": jnp.fft.irfft2(f), "T": T, "sx": input_dict["sx"] } )
        for i in range(macrosteps):
            f = update(f)
            savemat( f"timeseries/{i+1}.mat", {"f": jnp.fft.irfft2(f), "T": T, "sx": input_dict["sx"] } )

        savemat( f"data/RPO_candidate_{t}.mat", input_dict)
