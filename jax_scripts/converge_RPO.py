'''
Converge RPOs with ADAM
'''

import time
import jax
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
import lib.adam as adam
import lib.dictionaryIO as dictionaryIO

from scipy.io import savemat, loadmat

# If you want double precision, change JAX defaults
precision = jnp.float64
if(precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)



#Load state from a turbulent trajectory
turb_dict, param_dict = dictionaryIO.load_dicts("turb.npz")

#MATLAB indices I picked from visually inspecting the recurrence diagram.
idx = [54, 63]


#Get conditions for RPO guess
f = turb_dict['fs'][idx[0]-1,:,:,:]
f = jnp.fft.irfft2(f)

#Period
T = param_dict['dt'] * param_dict['ministeps'] * (idx[1] - idx[0])

#spatial shift
sx = 0.0

#number of timesteps 
steps = param_dict['ministeps'] * (idx[1] - idx[0])
steps = int(steps) #JAX complains otherwise
param_dict.update({ 'steps': steps } )

#Create a dictionary of optimizable field
input_dict = {"fields": f, "T": T, "sx": sx}

#Delete keys from the turbulent trajectory param_dict that we won't need anymore to avoid confusion
del param_dict['dt']
del param_dict['ministeps']


#Or, load dicts from a previous run
input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_264.npz")

#For some reason JAX complains that "steps" is not a constant unless I override is as an integer
param_dict['steps'] = int(param_dict['steps'])

print(f"using {param_dict['steps']} timesteps of type {type(param_dict['steps'])} ")



###############################
# Adjoint descent time
###############################

m, v = adam.init_adam(input_dict)
maxit = 10000000

#Define a function to compute the vlaue of the loss and the gradient simultaneously
#loss_fn = lambda input_dict: loss_functions.loss_RPO(input_dict, param_dict)
#grad_fn = jax.jit(jax.value_and_grad(loss_fn))


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
        dictionaryIO.save_dicts( f"data/adjoint_descent_{t}.npz", input_dict, param_dict )