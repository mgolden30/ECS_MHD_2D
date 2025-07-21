'''
Written July 18th 2025 by Matthew Golden

Version 1.0
Notes go here
'''

import jax.flatten_util
import lib.mhd_jax as mhd_jax
import time
import jax
import jax.numpy as jnp
import lib.dictionaryIO as dictionaryIO
import lib.loss_functions as loss_functions
import lib.adam as adam


from scipy.io import savemat, loadmat

precision = jnp.float64
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

#Read in a state
#input_dict, param_dict = dictionaryIO.load_dicts("newton/2.npz")
input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE2.npz")

#Check if this is a multishooting state
if "segments" in param_dict:
    print("Loaded multishooting state.")
else:
    segments = 8
    print(f"Loaded single shooting state. Creating a multishooting state with {segments} segments...")

    #Read grid resolution n 
    n = input_dict['fields'].shape[-1]

    #allocate memory 
    f = jnp.zeros([segments, 2, n, n], dtype=precision)

    #Set initial segment from single shooting data
    f = f.at[0,:,:,:].set(input_dict['fields'])

    #compute timestep
    dt = input_dict['T']/param_dict['steps']
    assert(param_dict['steps'] % segments == 0)
    ministeps = param_dict['steps']//segments
    param_dict.update({'segments': segments})

    @jax.jit
    def next_segment(f):
        #Integrate forward a bit
        g = jnp.fft.rfft2(f)
        g = mhd_jax.eark4(g, dt, ministeps, param_dict)

        #Translate in space
        g = jnp.exp( -1j * param_dict['kx'] * input_dict['sx'] / segments ) * g
        g = jnp.fft.irfft2(g)
        return g

    #Integrate forward
    for i in range(segments-1):
        print(i)
        g = next_segment(f[i,:,:,:])
        f = f.at[i+1,:,:,:].set(g)

    #Update the state dictionary to contain a set of multishooting points
    input_dict['fields'] = f

#How many checkpoints do we want to keep memory down?
param_dict.update({"checkpoints": 8})

#Do sanity checks + initialization of ministeps and miniministeps
assert(param_dict['steps'] % segments == 0)
ministeps = param_dict["steps"] // param_dict["segments"]
assert( ministeps % param_dict["checkpoints"] == 0 )
miniministeps = ministeps // param_dict["checkpoints"]

param_dict.update({"ministeps": ministeps, "miniministeps": miniministeps})

start = time.time()
loss = loss_functions.objective_RPO_multishooting( input_dict, param_dict )
stop = time.time()

print(f"Computed objective in {stop-start:.3e} seconds.")

for i in range(param_dict['segments']):
    diff = loss['fields'][i,:,:,:]
    print(f"max abs error in segment #{i} = {jnp.max(jnp.abs(diff)):.6f}")


#Take the 2-norm of the RPO objective
scalar_loss = lambda input_dict: jnp.linalg.norm( jax.flatten_util.ravel_pytree( loss_functions.objective_RPO_multishooting(input_dict, param_dict) )[0] )

#Define a gradient function

grad_fn = jax.jit(jax.value_and_grad(scalar_loss))
_, _ = grad_fn(input_dict)

start = time.time()
val, grad = grad_fn(input_dict)
stop = time.time()

print(f"Calculated val = {val:.3e} in {stop-start:.3f} seconds.")







###############################
# Adjoint descent time
###############################

m, v = adam.init_adam(input_dict)
maxit = 10000000

#Jit the update routine for ADAM to attempt some speedup
update_fn = jax.jit(adam.adam_update)

for t in range(maxit):
    start = time.time()
    loss, grad = grad_fn(input_dict)
    stop = time.time()
    walltime = stop-start

    lr = 1e-4
    input_dict, m, v = update_fn(input_dict, grad, m, v, t+1, lr=lr, beta1=0.9, beta2=0.999, eps=1e-6)

    #dealias
    f = input_dict['fields']
    f = jnp.fft.rfft2(f) * param_dict['mask']
    f = jnp.fft.irfft2(f)
    input_dict['fields'] = f

    print(f"{t}: loss = {loss}, walltime: {walltime}, T = {input_dict['T']}")
    
    if ( t % 8 == 0 ):
        dictionaryIO.save_dicts( f"data/adjoint_descent_{t}.npz", input_dict, param_dict )