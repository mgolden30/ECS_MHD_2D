'''
Perform Newton-GMRES to converge RPOs
'''

import time
import jax
import jax.flatten_util
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
import lib.adam as adam
from lib.linalg import gmres
import lib.dictionaryIO as dictionaryIO

###############################
# Construct numerical grid
###############################

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_40.npz")
input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_648.npz")
input_dict, param_dict = dictionaryIO.load_dicts("newton/2.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("Re40/RPO1.npz")


#Define the RPO objective function and compile it
objective = jax.jit( lambda input_dict: loss_functions.objective_RPO(input_dict, param_dict) )
f = objective(input_dict)

#Define the Jacobian action and compile it
jac = jax.jit( lambda primal, tangent: jax.jvp( objective, (primal,), (tangent,))[1] )
_ = jac( input_dict, f )

#Add phase conditions and compile it
jac_with_phase = jax.jit( lambda primal, tangent: loss_functions.add_phase_conditions( primal, tangent, jac(primal, tangent), param_dict) )
_ = jac_with_phase( input_dict, f )




######################################
# Newton-GMRES starts here
######################################

use_basic_gmres = True

maxit = 1024
inner = 256*2
outer = 1
damp  = 1.0
s_min = 1 #1e-2


for i in range(maxit):
    #Step 1: evaluate the objective function
    start = time.time()
    f = objective(input_dict)
    stop = time.time()
    f_walltime = stop - start

    #f is a dictionary. Turn it into a single vector for linear algebra
    f_vec, unravel_fn = jax.flatten_util.ravel_pytree(f)

    #Define a linear operator for GMRES
    #Do this every iteration since input_dict changes
    lin_op = lambda v:  jax.flatten_util.ravel_pytree(  jac_with_phase( input_dict, unravel_fn(v))  )[0]

    #Do GMRES
    start = time.time()
    step = gmres( lin_op, f_vec, inner, f_vec, s_min )
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