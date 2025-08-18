'''
The goal of this script is to use Newton-GMRES.
'''


import time
import jax
import jax.flatten_util
import jax.numpy as jnp

#import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
from lib.linalg import gmres
import lib.dictionaryIO as dictionaryIO
import lib.utils as utils


###############################
# Construct numerical grid
###############################

print(jax.devices())

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("temp_data/adjoint_descent/64.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re50/candidate4.npz")




##################
# NEWTON OPTIONS
##################

shooting_mode = "single_shooting" #"single_shooting" or "multi_shooting"
integrate_mode = "adaptive" #"fixed_timesteps" or "adaptive"
adaptive_dict = {
    "atol": 1e-4, #We make the timestep small enough that each step has max(abs(err)) < atol
    "checkpoints": 32, #How many times so we restart integration to preserve memory?
    "max_steps_per_checkpoint": 32 #How many steps do we take per timestep?
}
num_checkpoints = 32 #for fixed timestep integration. Modify the adaptive_dict for adaptive timestepping

use_transpose = True  #False solves Ax=b. True solves A^T A x = A^T b
s_min = 0    #What is the smallest singular value you are comfortable inverting. If s_min=0, you just compute the lstsq solution.
maxit = 1024 #Max iterations
inner = 16   #Krylov subspace dimension
outer = 1    #How many times should we restart GMRES? Only do restarts if you literally can't fit a larger Krylov subsapce in memory.

do_line_search = True #When we have a Newton step, should we do a line search in that direction?
default_damp  = 0.1 #if you don't do a line search, then damp the newton step with this

residual_stuck = False #Changing this to True will overwrite the initial condition with the final condition in an attempt to unstuck the residual. Results may vary


obj, param_dict = utils.choose_objective_fn( shooting_mode, integrate_mode, param_dict, num_checkpoints, adaptive_dict )

#print info
_ = obj(input_dict, param_dict)

#utility function to JIT the objective function and get a function for the JVP
obj, jac = utils.compile_objective_and_Jacobian( input_dict, param_dict, obj )

#Evaluate the objective
f = obj(input_dict)

#Use this value to overwrite the initial data
if residual_stuck:
    input_dict['fields'] = f['fields']/(1.0 + 1.0/input_dict['T']) + input_dict['fields']

#Define a simple lambda for flattening dictionaries 
flatten = lambda x: jax.flatten_util.ravel_pytree(x)[0]

#Similarly, define unflattening functions
_, unflatten_left  = jax.flatten_util.ravel_pytree(f)
_, unflatten_right = jax.flatten_util.ravel_pytree(input_dict)

def run_and_time( fn, x ):
    start = time.time()
    y = fn(x)
    stop = time.time()
    return y, stop-start

def relative_error_RPO( input_dict, f ):
    norm = lambda x: jnp.sqrt(jnp.sum( jnp.square(x)) )
    return norm(f["fields"]) / norm(input_dict["fields"])




######################################
# Newton-GMRES starts here
######################################

for i in range(maxit):
    #Evaluate the objective function
    f, f_walltime = run_and_time(obj, input_dict)

    #Turn f into a vector
    f_vec = flatten(f)

    #Define the Jacobian matrix acting on vectors, not dictionaries
    if not use_transpose:
        lin_op = jax.jit(lambda x: flatten(jac(input_dict, unflatten_right(x))))
        b = f_vec
    else:
        A = jax.jit(lambda x: flatten(jac(input_dict, unflatten_right(x))))
        _, jacT = jax.vjp( obj, input_dict, has_aux=False )
        A_T = jax.jit( lambda v: flatten( jacT(unflatten_left(v)) ) )
        #Compile it
        _ = A_T(f_vec)

        lin_op = lambda x: A_T(A(x))
        b, transpose_walltime = run_and_time( A_T, f_vec )
        if i == 0:
            #Print walltime of the transpose
            print(f"Transpose walltime = {transpose_walltime:.3f}")



    #Do GMRES
    start = time.time()
    step, gmres_residual = gmres(lin_op, b, inner, s_min, tol=1e-8, preconditioner_list=[], output_index=0 )
    stop = time.time()
    gmres_walltime = stop - start
    
    #update the input_dict
    x, unravel_fn = jax.flatten_util.ravel_pytree( input_dict )
    
    if do_line_search:
        damp = 1.0
        for _ in range(20):
            x_temp = x - damp * step
            temp_dict = unravel_fn(x_temp)
            f_temp = obj(temp_dict)
            f_temp_vec = jax.flatten_util.ravel_pytree(f_temp)[0]
            if ( jnp.linalg.norm(f_temp_vec) < jnp.linalg.norm(f_vec) ):
                x = x_temp
                break
            damp = damp/2
        input_dict = unravel_fn(x)
    else:
        damp = default_damp
        x = x - damp*step
        input_dict = unravel_fn(x)


    #Print all information after the GMRES step
    rel_err = relative_error_RPO(input_dict, f)
    print(f"Iteration {i}: rel_err={rel_err:.3e}, |f|={jnp.linalg.norm(f_vec):.3e}, fwall={f_walltime:.3f}, gmreswall={gmres_walltime:.3f}, gmres_rel_res={gmres_residual:.3e}, damp={damp:.3e}, T={input_dict['T']:.3e}, sx={input_dict['sx']:.3e}")

    #Dealias after every Newton step
    fields = input_dict['fields']
    fields = param_dict['mask'] * jnp.fft.rfft2(fields)
    fields = jnp.fft.irfft2(fields)
    input_dict['fields'] = fields

    dictionaryIO.save_dicts( f"temp_data/newton/{i}", input_dict, param_dict )
