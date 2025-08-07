'''
Written by Matthew Golden August 7th

Use Newton-Krylov methods to converge RPOs
'''




import time
import jax
import jax.flatten_util
import jax.numpy as jnp

#import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
from lib.linalg import adjoint_GMRES
import lib.dictionaryIO as dictionaryIO
import lib.preconditioners as precond
import lib.utils as utils

from scipy.io import savemat, loadmat

###############################
# Construct numerical grid
###############################

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_8.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE_multi.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re200/1_high_res.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("newton/50.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_6912.npz")

input_dict, param_dict = dictionaryIO.load_dicts("high_res.npz")

#mode = "multi_shooting"
mode = "single_shooting"

if mode == "single_shooting":
    print(param_dict["steps"])
    print(input_dict["fields"].shape)

    #define number of segements for memory checkpointing
    num_checkpoints = 32
    param_dict.update(  {"ministeps": int(param_dict["steps"]//num_checkpoints), "num_checkpoints": int(num_checkpoints)})

    #Define the RPO objective function
    obj = loss_functions.objective_RPO_with_checkpoints


if mode == "multi_shooting":
    #Define the RPO objective function
    obj = loss_functions.objective_RPO_multishooting






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
    norm = lambda x: jnp.sum( jnp.square(x) )
    return norm(f["fields"]) / norm(input_dict["fields"])



######################################
# Newton-GMRES starts here
######################################

maxit = 1024
inner = 32
outer = 1
damp  = 0.1

for i in range(maxit):
    #Evaluate the objective function
    f, f_walltime = run_and_time(objective, input_dict)

    #Compute the magnitude of the state vector
    print(f"relative error {relative_error_RPO(input_dict, f):.3e}")

    #Define a linear operator for GMRES
    #Do this every iteration since input_dict changes
    lin_op = lambda v: flat(jac(input_dict, unflat_right(v)))

    #Define a linear operator for the transpose
    _, jacT = jax.vjp( objective, input_dict, has_aux=False )
    lin_op_T = jax.jit( lambda v: jax.flatten_util.ravel_pytree( jacT(unravel_fn_left(v)) )[0] )

    #Do GMRES
    start = time.time()
    step = adjoint_GMRES( A=lin_op,  A_t=lin_op_T, b=f_vec, m=f_vec.size, n=input_vec.size, inner=inner, outer=outer, precond_left=[], x0_fn=lambda x,_: x, seed=0)
    stop = time.time()
    gmres_walltime = stop - start

    print(f"Iteration {i}: |f| = {jnp.linalg.norm(f_vec):.3e}, fwalltime = {f_walltime:.3f}, gmres time = {gmres_walltime:.3f}, period T = {input_dict['T']:.3e}")
    
    #update the input_dict
    x, unravel_fn = jax.flatten_util.ravel_pytree( input_dict )
    
    
    #Do a line search
    damp = 1.0
    for _ in range(20):
        x_temp = x - damp * step
        temp_dict = unravel_fn(x_temp)
        f_temp = objective(temp_dict)
        f_temp_vec = jax.flatten_util.ravel_pytree(f_temp)[0]

        print(f"Trying damp = {damp:.3e}")
        if ( jnp.linalg.norm(f_temp_vec) < jnp.linalg.norm(f_vec) ):
            x = x_temp
            break
        damp = damp/2
    input_dict = unravel_fn(x)
    
    #x = x - damp*step
    #input_dict = unravel_fn(x)
    

    #Dealias after every Newton step
    fields = input_dict['fields']
    fields = param_dict['mask'] * jnp.fft.rfft2(fields)
    fields = jnp.fft.irfft2(fields)
    input_dict['fields'] = fields

    dictionaryIO.save_dicts( f"newton/{i}", input_dict, param_dict )
