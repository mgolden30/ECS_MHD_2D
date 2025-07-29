'''
Perform Newton-GMRES to converge RPOs

In this variant, I would like to get the adjoint of the Jacobian working.
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

from scipy.io import savemat, loadmat

###############################
# Construct numerical grid
###############################

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

#input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_40.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_680.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE_multi.npz")
input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE5.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("newton/4.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_80.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("test.npz")

#mode = "multi_shooting"
mode = "single_shooting"

if mode == "single_shooting":
    #define number of segements for memory checkpointing
    num_checkpoints = 8
    param_dict.update(  {"ministeps": int(param_dict["steps"]//num_checkpoints), "num_checkpoints": int(num_checkpoints)})

    #Define the RPO objective function
    obj = loss_functions.objective_RPO_with_checkpoints

if mode == "multi_shooting":
    #Define the RPO objective function
    obj = loss_functions.objective_RPO_multishooting


#Use Floquet analysis to modift the objective function with phase conditions.
phase_conditions=True
if phase_conditions:
    data = loadmat("floquet.mat")
    
    #Compute eigenvalues and eigenvectors of R
    eigenval, eigenvec = jnp.linalg.eig( data['R'] )

    #Find the eigenvalues close to unity
    cutoff = 0.1
    marginal = jnp.abs( eigenval - 1) < cutoff

    #Isolate the margingal eigenvalues
    Q = eigenvec[:, marginal]
    Q = jnp.real( (1 + 0.4j) * Q )

    #Make tang a rectangular matrix
    tang = jnp.reshape( data['tang'], [data['R'].shape[0], -1] )
    Q = Q.transpose() @ tang
    
    #Update the loss function with orthogonality conditions
    obj = loss_functions.add_orthogonal_contraints( obj, param_dict, Q )




#Capture param_dict and JIT it
objective = jax.jit( lambda input_dict: obj(input_dict, param_dict) )


#Compile the objective function
f = objective(input_dict)

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

#Define the transpose of the Jacobian action
_, jacT = jax.vjp( objective, input_dict, has_aux=False )
jacT = jax.jit(jacT)
Jtf = jacT(f)

start = time.time()
Jtf = jacT(f)
stop = time.time()
walltime2 = stop - start

print(f"Evaluating objective: {walltime0:.3} seconds")
print(f"Evaluating Jacobian: {walltime1:.3} seconds")
print(f"Evaluating Jacobian transpose: {walltime2:.3} seconds")

############################
# Compute a preconditioner
############################
#M1 = precond.diagonal_preconditioner_fourier( input_dict, jac, k=8, batch=16 )
#M1 = precond.diagonal_preconditioner_spatial(input_dict, param_dict, jac, k=16, batch=16)
#M1 = precond.floquet_preconditioner( "floquet.mat", epsilon=1.0 )




######################################
# Newton-GMRES starts here
######################################

maxit = 1024
inner = 32
outer = 1

for i in range(maxit):
    #Evaluate the objective function
    start = time.time()
    f = objective(input_dict)
    stop = time.time()
    f_walltime = stop - start

    #f and input_dict are dictionaries. Flatten them into vectors for linear algebra
    f_vec,     unravel_fn_left  = jax.flatten_util.ravel_pytree(f)
    input_vec, unravel_fn_right = jax.flatten_util.ravel_pytree(input_dict)


    #Compute the magnitude of the state vector
    s_vec, _ = jax.flatten_util.ravel_pytree( {"fields": input_dict['fields']} )
    print(f"relative error {jnp.linalg.norm(f_vec) / jnp.linalg.norm(s_vec):.3e}")

    #Define a linear operator for GMRES
    #Do this every iteration since input_dict changes
    lin_op = lambda v:  jax.flatten_util.ravel_pytree(  jac( input_dict, unravel_fn_right(v))  )[0]

    #Define a linear operator for the transpose
    _, jacT = jax.vjp( objective, input_dict, has_aux=False )
    lin_op_T = jax.jit( lambda v: jax.flatten_util.ravel_pytree( jacT(unravel_fn_left(v)) )[0] )

    #Do GMRES
    start = time.time()
    step = adjoint_GMRES( lin_op, lin_op_T, f_vec, f_vec.size, input_vec.size, inner, outer=outer, preconditioner_list=[])
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

    #x = x - 0.01*step
    #input_dict = unravel_fn(x)


    #Dealias after every Newton step
    fields = input_dict['fields']
    fields = param_dict['mask'] * jnp.fft.rfft2(fields)
    fields = jnp.fft.irfft2(fields)
    input_dict['fields'] = fields

    dictionaryIO.save_dicts( f"newton/{i}", input_dict, param_dict )
