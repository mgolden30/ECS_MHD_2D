'''
Written by Matthew Golden sometime in 2025.

PURPOSE
---------
The purpose of this script is to converge Relative Periodic Orbits (RPOs) to the 
2D Magnetohydrodynamic equations with gradient descent.

CHANGES
--------
Version 0.1 ()

'''

import time
import jax
import jax.numpy as jnp

import os

import lib.loss_functions as loss_functions
import lib.adam as adam
import lib.dictionaryIO as dictionaryIO
import lib.utils as utils

os.makedirs( "temp_data/adjoint_descent", exist_ok=True)

###############################
# PARAMETERS OF THIS SCRIPT.
###############################

# ideally, you do not have to change anything below this section
precision = jnp.float64
filename = "temp_data/turb.npz" #Set to "turb.npz" for a new state or "data/adjoint_descent_8.npz" for example if you want to restart optimization for an old state
#filename = "temp_data/adjoint_descent/512.npz" 
idx = [109, 138] #If filename == "turb.npz", then these will determine the initial guess of the RPO from turbulence 
lr = 1e-2 #Learning rate of ADAM
maxit = 16*1024 #Maximum number of ADAM steps
save_every = 64 #Save the fluid state after this many ADAM steps.
#The current function loss_functions.loss_RPO uses adaptive timestepping, which has benefits and drawbacks.
#Define a dictionary of parameters that you can tweak here.
adaptive_dict = {
    "atol": 1e-4, #We make the timestep small enough that each step has max(abs(err)) < atol
    "checkpoints": 128, #How many times so we restart integration to preserve memory?
    "max_steps_per_checkpoint": 32 #How many steps do we take per timestep?
}

#specify which kind of time integration you want to do.
mode = "Lawson_RK4"


if(precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

if filename == "temp_data/turb.npz":
    print(f"Creating new RPO guess from {filename}...")
    turb_dict,  param_dict = dictionaryIO.load_dicts(filename)
    input_dict, param_dict = utils.create_state_from_turb(turb_dict, idx, param_dict)
elif os.path.isfile(filename):
    print(f"Loading existing state from {filename}...")
    input_dict, param_dict = dictionaryIO.load_dicts(filename)
else:
    print(f"Error: Your specified filename = {filename} could not be opened.")
    print("Please try again...")
    exit()

#For some reason JAX complains that "steps" is not a constant unless I override is as an integer
param_dict['steps'] = int(param_dict['steps'])

#For fixed timestep solvers, we need to define parameters for checkpointing.
ministeps = 32
assert( param_dict['steps'] % ministeps == 0 )
param_dict.update({"ministeps": ministeps, "num_checkpoints": param_dict['steps']//ministeps })

print(f"using {param_dict['steps']} timesteps of type {type(param_dict['steps'])} ")



###############################
# Begin Adjoint Descent
###############################


#initialize ADAM variables and JIT the ADAM step.
m, v = adam.init_adam(input_dict)
update_fn = jax.jit(adam.adam_update)


#Define a function to compute the vlaue of the loss and the gradient simultaneously
loss_fn = lambda input_dict: loss_functions.loss_RPO( input_dict, param_dict, mode )
grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

for t in range(maxit):
    #Compute the loss function and gradient
    start = time.time()
    (loss, info), grad = grad_fn(input_dict)
    stop = time.time()
    walltime = stop-start

    #Complain if the adaptive timestepping fails to complete integration.
    if isinstance(info, dict) and info.get("completed") is False:
        print("ERROR: did not complete integration...")
        print("Integration info for debugging:")
        print(info)
        exit()
    
    #Apply the ADAM update
    input_dict, m, v = update_fn(input_dict, grad, m, v, t+1, lr=lr, beta1=0.9, beta2=0.999, eps=1e-6)

    #dealias fields
    f = input_dict['fields']
    f = jnp.fft.rfft2(f) * param_dict['mask']
    f = jnp.fft.irfft2(f)
    input_dict['fields'] = f

    #Print diagnostics
    #print(f"{t}: loss={loss:.6f}, walltime={walltime:.3f}, T={input_dict['T']:.3f}, sx={input_dict['sx']:.3f}, completed={info['completed']}, fevals={info['fevals']}, accepted={info['accepted']}, rejected={info['rejected']}")
    print(f"{t}: loss={loss:.6f}, walltime={walltime:.3f}, T={input_dict['T']:.3f}, sx={input_dict['sx']:.3f}")

    #Save the state every so often
    if ( t % save_every == 0 ):
        remove_bloat = lambda param_dict : dictionaryIO.remove_grid_information(param_dict)
        dictionaryIO.save_dicts( f"temp_data/adjoint_descent/{t}.npz", input_dict, remove_bloat(param_dict) )
