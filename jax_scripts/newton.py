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

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_128.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE_multi.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re200/1_high_res.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("high_res.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("temp.npz")
input_dict, param_dict = dictionaryIO.load_dicts("newton/2.npz")
input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_12608.npz")




##################
# NEWTON OPTIONS
##################

shooting_mode = "single_shooting" #"single_shooting" or "multi_shooting"
integrate_mode = "adaptive" #"fixed_timesteps" or "adaptive"
use_transpose = "no" #"yes" or "no". "no" solves Ax=b. "yes" solves A^T A x = A^T b
adaptive_dict = {
    "atol": 1e-4, #We make the timestep small enough that each step has max(abs(err)) < atol
    "checkpoints": 32, #How many times so we restart integration to preserve memory?
    "max_steps_per_checkpoint": 32 #How many steps do we take per timestep?
}
num_checkpoints = 32 #for fixed timestep integration







obj, param_dict = utils.choose_objective_fn( shooting_mode, integrate_mode, param_dict, num_checkpoints, adaptive_dict )

#utility function to JIT the objective function and get a function for the JVP
obj, jac = utils.compile_objective_and_Jacobian( input_dict, param_dict, obj )

#Evaluate the objective
f = obj(input_dict)

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

maxit = 1024
inner = 128
outer = 1
damp  = 0.1

for i in range(maxit):
    #Evaluate the objective function
    f, f_walltime = run_and_time(obj, input_dict)

    #Compute the magnitude of the state vector
    print(f"relative error {relative_error_RPO(input_dict, f):.3e}")

    #Turn f into a vector
    f_vec = flatten(f)

    #Define the Jacobian matrix acting on vectors, not dictionaries
    lin_op = jax.jit(lambda x: flatten(jac(input_dict, unflatten_right(x))))

    #Do GMRES
    start = time.time()
    s_min = 1
    step = gmres(lin_op, f_vec, inner, s_min, tol=1e-8, preconditioner_list=[], output_index=0 )
    stop = time.time()
    gmres_walltime = stop - start

    print(f"Iteration {i}: |f| = {jnp.linalg.norm(f_vec):.3e}, fwalltime = {f_walltime:.3f}, gmres time = {gmres_walltime:.3f}, period T = {input_dict['T']:.3e}")
    
    #update the input_dict
    x, unravel_fn = jax.flatten_util.ravel_pytree( input_dict )
    
    #'''
    #Do a line search
    damp = 1.0
    for _ in range(20):
        x_temp = x - damp * step
        temp_dict = unravel_fn(x_temp)
        f_temp = obj(temp_dict)
        f_temp_vec = jax.flatten_util.ravel_pytree(f_temp)[0]

        print(f"Trying damp = {damp:.3e}")
        if ( jnp.linalg.norm(f_temp_vec) < jnp.linalg.norm(f_vec) ):
            x = x_temp
            break
        damp = damp/2
    input_dict = unravel_fn(x)
    #'''

    #x = x - damp*step
    #input_dict = unravel_fn(x)

    #Dealias after every Newton step
    fields = input_dict['fields']
    fields = param_dict['mask'] * jnp.fft.rfft2(fields)
    fields = jnp.fft.irfft2(fields)
    input_dict['fields'] = fields

    dictionaryIO.save_dicts( f"newton/{i}", input_dict, param_dict )
