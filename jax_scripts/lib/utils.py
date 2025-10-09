'''
This will be the junkyard of my code. If a function is useful, but I don't think you should care much how it works,
it goes here.
'''

import jax
import jax.numpy as jnp

import lib.loss_functions as loss_functions
import lib.symmetry as symm



def create_state_from_turb( turb_dict, idx, param_dict ):
    #Get conditions for RPO guess
    f1 = turb_dict['fs'][idx[0]-1,:,:,:]
    f2 = turb_dict['fs'][idx[1]-1,:,:,:]
    
    g = symm.find_optimal_symmetry(f1,f2,param_dict,nx=64)
    
    #Return the state in real space
    f = jnp.fft.irfft2(f1)

    #Period
    T = param_dict['dt'] * param_dict['ministeps'] * (idx[1] - idx[0])

    #number of timesteps 
    steps = param_dict['ministeps'] * (idx[1] - idx[0])
    steps = int(steps) #JAX complains if we do not cast steps

    param_dict.update({ 'steps': steps, 'shift_reflect_ny': g[1], 'rot': g[2] } )

    #Create a dictionary of optimizable field
    input_dict = {"fields": f, "T": T, "sx": g[0]}

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





def choose_objective_fn( shooting_mode, integrate_mode, param_dict, num_checkpoints, adaptive_dict ):
    #Check that all options are within their allowed values
    assert shooting_mode in {"single_shooting", "multi_shooting"}
    assert integrate_mode in {"adaptive", "fixed_timesteps"}


    if shooting_mode == "single_shooting" and integrate_mode == "fixed_timesteps":
        print(f"Choosing single shooting with fixed timesteps:")
        print(f"steps = {param_dict['steps']}")
        print(f"num_checkpoints = {num_checkpoints}")
        #define number of segements for memory checkpointing
        param_dict.update(  {"ministeps": int(param_dict["steps"]//num_checkpoints), "num_checkpoints": int(num_checkpoints)})
        #Define the RPO objective function
        obj = loss_functions.objective_RPO_with_checkpoints

    if shooting_mode == "single_shooting" and integrate_mode == "adaptive":
        print(f"Choosing single shooting with adaptive timestepping:")
        obj = lambda input_dict, param_dict: loss_functions.objective_RPO_adaptive( input_dict, param_dict, adaptive_dict )

    if shooting_mode == "multi_shooting":
        print(f"ERROR: multishooting is experimental and not quite implemented. Yell at Matt.")
        exit()
        #Define the RPO objective function
        #obj = loss_functions.objective_RPO_multishooting

    return obj, param_dict
