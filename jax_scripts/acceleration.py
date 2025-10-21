'''
Written by Matthew Golden some time in 2025

PURPOSE:
This is a modification of newton.py to use Geodesic Acceleration as described in 
Improvements to the Levenberg-Marquardt algorithm for nonlinear least-squares minimization
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

os.makedirs( "temp_data/geodesic", exist_ok=True)

###############################
# Construct numerical grid
###############################

print(jax.devices())

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

#input_dict, param_dict = dictionaryIO.load_dicts("temp_data/adjoint_descent/640.npz")
input_dict, param_dict = dictionaryIO.load_dicts("temp_data/newton/8.npz")


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
@jax.jit
def newton_gmres_update(i, input_dict):
    '''
    Attempt to JIT compile the entire Newton-GMRES update process.
    This is more of a feat of strength than a strict requirement.
    '''
    
    #Evaluate the objective function
    f, f_walltime = run_and_time(obj, input_dict)

    #Turn f into a vector
    b = -flatten(f)

    #Define the Jacobian matrix acting on vectors, not dictionaries
    J = lambda x: flatten(jac(input_dict, unflatten_right(x)))
    
    #Define the transpose of the Jacobian
    _, jacT = jax.vjp( obj, input_dict, has_aux=False )
    J_T = lambda v: flatten( jacT(unflatten_left(v)) )
    
    #Solve for the first order step
    step1, gmres_residual = gmres(J, b, inner, outer, preconditioner_list=[J_T])
    
    #Use this first order step to compute second derivative information
    obj2 = lambda x : flatten(obj(unflatten_right(x)))
    #In the paper they call this second derivative tensor K
    K = lambda v : jax.jvp( lambda x : jax.jvp(obj2, primals=(x,), tangents=(v,))[1], primals=(flatten(input_dict),), tangents=(v,) )[1]

    r = -K(step1)/2

    #Perform GMRES and time it
    step2, gmres_residual = gmres(J, r, inner, outer, preconditioner_list=[J_T])

    #update the input_dict
    x, unravel_fn = jax.flatten_util.ravel_pytree( input_dict )

    #Combine the steps
    x = x + step1 + step2

    input_dict = unravel_fn(x)

    gmres_walltime = 1.234
    damp = 1.0

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
    input_dict = newton_gmres_update(i, input_dict)
    dictionaryIO.save_dicts( f"temp_data/geodesic/{i}", input_dict, param_dict )