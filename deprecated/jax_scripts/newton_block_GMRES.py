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
from lib.linalg import gmres, block_gmres
import lib.dictionaryIO as dictionaryIO

from scipy.io import savemat, loadmat

###############################
# Construct numerical grid
###############################


precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)


input_dict, param_dict = dictionaryIO.load_dict("data/adjoint_descent_64.npz")


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
inner = 8
outer = 1
damp  = 1.0




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

    '''
    f1 = lambda tangents: jac_with_phase(input_dict, tangents)
    f2 = lambda x: jax.flatten_util.ravel_pytree(x)[0]

    #lambda that acts on a vector and returns a vector. We handle the dict <-> vector translation
    vector_jac = lambda vec: f2(f1(unravel_fn(vec)))
    '''

    '''
    batched_jac = jax.vmap( vector_jac, in_axes=1, out_axes=1 )

    batch_size = 4
    B = jnp.zeros((f_vec.shape[0], batch_size))

    #Let's generate a realistic subspace
    B = B.at[:,0].set( f_vec )
    fields = input_dict['fields']
    fields = jnp.fft.rfft2(fields)
    dfdx = 1j * param_dict['kx'] * fields
    dfdt = mhd_jax.state_vel(fields, param_dict, include_dissipation=True)
    dfdx = jnp.fft.irfft2(dfdx)
    dfdt = jnp.fft.irfft2(dfdt)

    B = B.at[:,1].set( f2({'fields': dfdt, 'T': 0, 'sx':0}) )
    B = B.at[:,2].set( f2({'fields': dfdx, 'T': 0, 'sx':0}) )

    #Lastly, adjoint descent direction
    B = B.at[:,3].set( jax.flatten_util.ravel_pytree( grad_fn(input_dict ) )[0] )


    m = 32*2
    start = time.time()
    step = block_gmres( batched_jac, f_vec, m, B, tol=1e-8, iteration=i )
    stop = time.time()
    '''

    step

    print(f"Iteration {i}: |f| = {jnp.linalg.norm(f_vec)}, fwalltime = {fwalltime}, gmres time = {stop - start}")
    
    #update the input_dict
    x, unravel_fn = jax.flatten_util.ravel_pytree( input_dict )
    x = x - damp * step
    input_dict = unravel_fn(x)

    #Dealias after every Newton step
    fields = input_dict['fields']
    k = jnp.fft.fftfreq(n, d=1/n, dtype=jnp.float64)
    kx = jnp.reshape(k,          [-1, 1])
    ky = jnp.reshape(k[:n//2+1], [1, -1])
    hypermask = (jnp.abs(kx) < n/4) & (jnp.abs(ky) < n/4)
    fields = hypermask * jnp.fft.rfft2(fields)
    fields = jnp.fft.irfft2(fields)
    input_dict['fields'] = fields

    save_jax_dict(f"newton/{i}.bin", input_dict)