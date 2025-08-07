'''
This will be the junkyard of my code. If a function is useful, but I don't think you should care much how it works,
it goes here.
'''

import jax
import jax.numpy as jnp

def create_state_from_turb( turb_dict, idx, param_dict ):
    #Get conditions for RPO guess
    f = turb_dict['fs'][idx[0]-1,:,:,:]
    f = jnp.fft.irfft2(f)

    #Period
    T = param_dict['dt'] * param_dict['ministeps'] * (idx[1] - idx[0])

    #spatial shift
    #Eventually do a search for a good intial guess
    sx = 0.0

    #number of timesteps 
    steps = param_dict['ministeps'] * (idx[1] - idx[0])
    steps = int(steps) #JAX complains if we do not cast steps

    param_dict.update({ 'steps': steps } )

    #Create a dictionary of optimizable field
    input_dict = {"fields": f, "T": T, "sx": sx}

    #Delete keys from the turbulent trajectory param_dict that we won't need anymore to avoid confusion
    del param_dict['dt']
    del param_dict['ministeps']

    return input_dict, param_dict


def compile_objective_and_Jacobian( input_dict, param_dict, obj ):
    
    #Capture param_dict and JIT the objective function
    objective = jax.jit( lambda input_dict: obj(input_dict, param_dict) )

    #Compile the objective function.
    f = objective(input_dict)

    import time
    start = time.time()
    f = objective(input_dict)
    stop = time.time()
    walltime0 = stop - start

    #Define the Jacobian action and compile it
    jac = jax.jit( lambda primal, tangent: jax.jvp( objective, (primal,), (tangent,))[1] )
    _ = jac( input_dict, input_dict )

    start = time.time()
    Jf = jac( input_dict, input_dict )
    stop = time.time()
    walltime1 = stop - start

    print(f"Evaluating objective: {walltime0:.3} seconds")
    print(f"Evaluating Jacobian: {walltime1:.3} seconds")
    #print(f"Evaluating Jacobian transpose: {walltime2:.3} seconds")
    return objective, jac
