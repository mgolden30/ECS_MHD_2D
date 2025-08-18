'''
Newton-Raphson iteration, but with an adjoint based method instead of GMRES.
'''

import time
import jax
import jax.flatten_util
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
import lib.adam as adam
from lib.linalg import adjoint_GMRES
import lib.dictionaryIO as dictionaryIO

###############################
# Construct numerical grid
###############################


precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)


input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_40.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("newton/46.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("Re40/RPO1.npz")

print(f"This RPO candidate is using {param_dict['steps']} steps")


num_checkpoints = 32
param_dict['num_checkpoints'] = jnp.array( num_checkpoints )
param_dict['ministeps'] = param_dict['steps'] // num_checkpoints

#Define the RPO objective function
objective = jax.jit( lambda input_dict: loss_functions.objective_RPO_with_checkpoints(input_dict, param_dict) )

#Define the Jacobian action and compile it
jac = jax.jit( lambda primal, tangent: jax.jvp( objective, (primal,), (tangent,))[1] )

def run_and_time( fn, *args, **kwargs ):
    start = time.time()
    result = fn(*args, **kwargs)
    end = time.time()
    walltime = end - start
    return result, walltime

######################################
# Newton-GMRES starts here
######################################

maxit = 1024
inner = 1
outer = 1
damp  = 0.1

for i in range(maxit):

    #Evaluate the objective function, which will return a dictionary
    f_dict, f_walltime = run_and_time( objective, input_dict )

    # Both input_dict and f_dict are dictionaries. Turn these into flat 1D vectors
    # So we can do linear algebra.
    f, unravel_f = jax.flatten_util.ravel_pytree(f_dict)
    _, unravel_x = jax.flatten_util.ravel_pytree(input_dict)

    print( f"max(abs(f)) = {jnp.max(jnp.abs(f))}" )

    #Define a linear operator for GMRES
    lin_op = lambda v:  jax.flatten_util.ravel_pytree( jac( input_dict, unravel_x(v)) )[0]

    #Define first the Adjoint operator   
    adjoint = jax.jit(lambda u: jax.vjp(objective, input_dict)[1](u)[0])

    #Wrap the adjoint action with flattening/unflattening
    lin_op_t = lambda u: jax.flatten_util.ravel_pytree( adjoint( unravel_f(u) ) )[0]

    print(unravel_f(f))
    output = lin_op_t( f )
    print(output)
    exit()

    #Do GMRES
    start = time.time()
    step = adjoint_GMRES( lin_op, lin_op_t, f_vec, f_vec.size, f_vec.size, inner )
    stop = time.time()
    gmres_walltime = stop - start

    print(f"Iteration {i}: |f| = {jnp.linalg.norm(f_vec):.3e}, fwalltime = {f_walltime:.3f}, gmres time = {gmres_walltime:.3f}, period T = {input_dict['T']:.3e}")
    
    #update the input_dict
    x, unravel_fn = jax.flatten_util.ravel_pytree( input_dict )
    x = x - damp * step
    input_dict = unravel_fn(x)

    #Dealias after every Newton step
    fields = input_dict['fields']
    #k = jnp.fft.fftfreq(n, d=1/n, dtype=jnp.float64)
    #kx = jnp.reshape(k,          [-1, 1])
    #ky = jnp.reshape(k[:n//2+1], [1, -1])
    #hypermask = (jnp.abs(kx) < n/4) & (jnp.abs(ky) < n/4)
    fields = param_dict['mask'] * jnp.fft.rfft2(fields)
    fields = jnp.fft.irfft2(fields)
    input_dict['fields'] = fields

    dictionaryIO.save_dicts( f"newton/{i}", input_dict, param_dict )