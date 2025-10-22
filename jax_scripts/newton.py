'''
Written by Matthew Golden some time in 2025

PURPOSE:
The goal of this script is to use Newton-GMRES and variants to converge periodic solutions to the MHD equations.
The Jacobian action is computed with autodiff via jax.jvp(). This allows us to get quite creative in the definition of 
the loss function.

NOTES:
Autodiff (and lessons learned in memory management) enable us to compute the adjoint (effectively the transpose) of the Jacobian.
This is equivalent to an adjoint solver. Instead of solving for the Newton step Js=-f, we can solve for the Gauss-Newton step
(J^T J)s = -J^Tf. In my experience, this requires significantly fewer Krylov dimensions and allows much larger step sizes. 
This is, as with any first order method, a risk of falling into local minima.
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

from jax.experimental import io_callback

import os

os.makedirs( "temp_data/newton", exist_ok=True)

###############################
# Construct numerical grid
###############################

print(jax.devices())

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("temp_data/adjoint_descent/4672.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("temp_data/newton/1.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re40/TW2.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("temp_data/newton/1.npz")


legacy_format = False
if legacy_format:
    ministeps = 32
    assert( param_dict['steps'] % ministeps == 0 )
    param_dict.update({"ministeps": ministeps, "num_checkpoints": param_dict['steps']//ministeps })
    param_dict.update({'shift_reflect_ny': 0, 'rot': False})
    input_dict['sx'] = -input_dict['sx']

#param_dict['forcing_str'] = "lambda x,y : -4*jnp.cos(4*y)"
param_dict = dictionaryIO.recompute_grid_information(input_dict, param_dict)



##################
# NEWTON OPTIONS
##################
#mode = "RK4"
#mode = "Lawson_RK4"
#mode = "Lawson_RK6"
mode = "TDRK4"
#mode = "Lawson_RK43"

#For adaptive timestepping, modify parameters here
if mode == "Lawson_RK43":
    adaptive_dict = {
    "atol": 1e-5, #We make the timestep small enough that each step has max(abs(err)) < atol
    "checkpoints": 128, #How many times so we restart integration to preserve memory?
    "max_steps_per_checkpoint": 32 #How many steps do we take per timestep?
    }
    #Add this dictionary as an element of param_dict
    param_dict["adaptive_dict"] = adaptive_dict


use_transpose = True  #False solves Ax=b. True solves A^T A x = A^T b
s_min = 0.0  #What is the smallest singular value you are comfortable inverting. If s_min=0, you just compute the lstsq solution.
maxit = 1024 #Max iterations
inner = 16   #Krylov subspace dimension  
outer = 1    #How many times should we restart GMRES? Only do restarts if you literally can't fit a larger Krylov subsapce in memory.

do_line_search = True #When we have a Newton step, should we do a line search in that direction?
default_damp  = 0.1 #if you don't do a line search, then damp the newton step with this

#obj, param_dict = utils.choose_objective_fn( shooting_mode, integrate_mode, param_dict, num_checkpoints, adaptive_dict )
obj = lambda input_dict, param_dict : loss_functions.objective_RPO(input_dict, param_dict, mode)

#print info
_ = obj(input_dict, param_dict)

#utility function to JIT the objective function and get a function for the JVP
obj, jac = utils.compile_objective_and_Jacobian( input_dict, param_dict, obj )

#Evaluate the objective
f = obj(input_dict)

#Define a simple lambda for flattening dictionaries 
flatten = lambda x: jax.flatten_util.ravel_pytree(x)[0]

#Similarly, define unflattening functions
_, unflatten_left  = jax.flatten_util.ravel_pytree(f)
_, unflatten_right = jax.flatten_util.ravel_pytree(input_dict)




def run_and_time( f, x ):
    '''
    JAX has lazy evaluation, which makes benchmarking a huge pain.
    This runs a function f(x) and returns the output and walltime.
    
    PARAMETERS
    ----------
    f : callable
        the function you want to evaluate
    x : array
        Argument of the function.
    '''

    #io_callback requires a shape struct for the returned tensor
    shape = jax.ShapeDtypeStruct( dtype=jnp.float64, shape=[] )
    start = io_callback(lambda _ : time.time(), shape, None)
    #x = x.at[0].set(x[0] + 0.0*start ) #trick lazy evaluation into thinking b depends on start
    wrapper = lambda _ : f(x)
    y = wrapper(start)
    stop = io_callback(lambda _ : time.time(), shape, y) #stop depends on output of the function
    walltime = stop - start
    #start = time.time()
    #y = fn(x)
    #stop = time.time()
    return y, walltime

def relative_error_RPO( input_dict, f ):
    norm = lambda x: jnp.sqrt(jnp.sum( jnp.square(x)) )
    return norm(f["fields"]) / norm(input_dict["fields"])




######################################
# Newton-GMRES starts here
######################################
@jax.jit
def newton_gmres_update(i, input_dict):
    '''
    Attempt to JIT compile the entire Newton-GMRES update process.
    This is more of a feat of strength than a strict requirement.
    '''
    
    #Evaluate the objective function
    f, f_walltime = run_and_time(obj, input_dict)

    #Turn f into a vector
    b = flatten(f)

    #Define the Jacobian matrix acting on vectors, not dictionaries
    A = lambda x: flatten(jac(input_dict, unflatten_right(x)))
    precond = [] #no preconditioners
    if use_transpose:
        _, jacT = jax.vjp( obj, input_dict, has_aux=False )
        A_T = lambda v: flatten( jacT(unflatten_left(v)) )
        precond = [A_T] #use the transpose (adjoint) as a preconditioner

    #Perform GMRES and time it
    '''
    shape = jax.ShapeDtypeStruct( dtype=jnp.float64, shape=[] )
    start = io_callback(lambda _ : time.time(), shape, None)
    b = b.at[0].set(b[0] + 0.0 *start ) #trick lazy evaluation into thinking b depends on start
    step, gmres_residual = gmres(A, b, inner, outer, preconditioner_list=precond)
    stop = io_callback(lambda _ : time.time(), shape, gmres_residual) #stop depends on gmres residual
    gmres_walltime = stop-start
    '''
    gmres_fn = lambda b : gmres(A, b, inner, outer, preconditioner_list=precond)
    (step, gmres_residual), gmres_walltime = run_and_time( gmres_fn, b )

    #update the input_dict
    x, unravel_fn = jax.flatten_util.ravel_pytree( input_dict )

    if do_line_search:
        x, damp = utils.line_search_unravel(x, step, obj, unravel_fn, b, max_iters=20)
    else:
        damp = default_damp
        x = x - damp*step
    input_dict = unravel_fn(x)

    #Print all information after the GMRES step
    rel_err = relative_error_RPO(input_dict, f)
    #jax.debug.print(f"Iteration {i}: rel_err={rel_err:.3e}, |f|={jnp.linalg.norm(b):.3e}, fwall={f_walltime:.3f}, gmreswall={gmres_walltime:.3f}, gmres_rel_res={gmres_residual:.3e}, damp={damp:.3e}, T={input_dict['T']:.3e}, sx={input_dict['sx']:.3e}")
    jax.debug.print(
        "Iteration {}: rel_err={:.3e}, |f|={:.3e}, fwall={:.3f}, gmreswall={:.3f}, gmres_rel_res={:.3e}, damp={:.3e}, T={:.3e}, sx={:.3e}",
        i,
        rel_err,
        jnp.linalg.norm(b),
        f_walltime,
        gmres_walltime,
        gmres_residual,
        damp,
        input_dict["T"],
        input_dict["sx"],   
    )
    
    #Dealias after every Newton step
    fields = input_dict['fields']
    fields = param_dict['mask'] * jnp.fft.rfft2(fields)
    fields = jnp.fft.irfft2(fields)
    input_dict['fields'] = fields
    return input_dict

for i in range(maxit):
    #Peform the newton step
    input_dict = newton_gmres_update(i, input_dict)

    #Save the state out
    remove_bloat = lambda param_dict : dictionaryIO.remove_grid_information(param_dict)
    dictionaryIO.save_dicts( f"temp_data/newton/{i}", input_dict, remove_bloat(param_dict) )