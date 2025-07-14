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
from lib.super_tensor_gmres import super_tensor_gmres
import lib.dictionaryIO as dictionaryIO
import lib.preconditioners as precond


###############################
# Construct numerical grid
###############################

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_40.npz")
input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_792.npz")
input_dict, param_dict = dictionaryIO.load_dicts("newton/20.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE.npz")
input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE2.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("newton/1.npz")


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



#For Tensor-GMRES, we need 
def second_directional_derivative_fn(f):
    """
    Returns a function that evaluates H_f(x)[v, v] for given x and v.
    """
    return lambda x, v: jax.jvp(
        lambda x_: jax.jvp(f, (x_,), (v,))[1],
        (x,),
        (v,)
    )[1]

hessian_fn = second_directional_derivative_fn(objective)
hessian_fn = jax.jit(hessian_fn)
_ = hessian_fn(input_dict, f)




######################################
# Newton-GMRES starts here
######################################

maxit = 1024
inner = 64
outer = 1
s_min = 0.1

damp = 1.0

for i in range(maxit):
    #Evaluate the objective function
    start = time.time()
    f = objective(input_dict)
    stop = time.time()
    f_walltime = stop - start

    #f is a dictionary. Turn it into a single vector for linear algebra
    f_vec, unravel_fn = jax.flatten_util.ravel_pytree(f)

    #Compute the magnitude of the state vector
    s_vec, _ = jax.flatten_util.ravel_pytree( {"fields": input_dict['fields']} )

    print(f"relative errror is {jnp.linalg.norm(f_vec) / jnp.linalg.norm(s_vec)}")
    #exit() 

    #Define a linear operator for GMRES
    lin_op = lambda v:  jax.flatten_util.ravel_pytree(  jac_with_phase( input_dict, unravel_fn(v))  )[0]

    #Define a Hessian function
    hess = lambda v: jax.flatten_util.ravel_pytree( hessian_fn(input_dict, unravel_fn(v) ) )[0]

    #Do tensor_GMRES
    start = time.time()
    step = super_tensor_gmres( jac=lin_op, f0=f_vec, m=inner, s_min=s_min, hessian_fn=hess )
    stop = time.time()
    gmres_walltime = stop - start

    print(f"Iteration {i}: |f| = {jnp.linalg.norm(f_vec):.3e}, fwalltime = {f_walltime:.3f}, gmres time = {gmres_walltime:.3f}, period T = {input_dict['T']:.3e}")
    
    #update the input_dict
    x, unravel_fn = jax.flatten_util.ravel_pytree( input_dict )
    
    #Tensor GMRES is meant to be incremented like this
    #x = x + damp*step

    #Do a line search for the best step that actually decreases the cost function
    max_attempts = 20
    damp = 1.0
    for j in range(max_attempts):
        x_trial = x + damp*step
        
        f_trial = f = objective(unravel_fn(x_trial))
        f_vec_trial = jax.flatten_util.ravel_pytree(f_trial)[0]

        if jnp.linalg.norm(f_vec_trial) < jnp.linalg.norm(f_vec):
            x = x_trial
            print(f"damp = {damp}")
            break
        damp = damp/2


    #Turn back to a dictionary
    input_dict = unravel_fn(x)

    #Dealias after every Newton step
    fields = input_dict['fields']
    fields = param_dict['mask'] * jnp.fft.rfft2(fields)
    fields = jnp.fft.irfft2(fields)
    input_dict['fields'] = fields

    dictionaryIO.save_dicts( f"newton/{i}", input_dict, param_dict )