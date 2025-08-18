'''
This script will be my main convergence script for Newton-Raphson iteration for RPOs
'''

import time
import jax.numpy as jnp
import jax

import mhd_jax_v2 as mhd_jax 

from scipy.io import savemat, loadmat

my_dict = loadmat("best.mat")

#Need to define fields, T, and sx
input_dict = {'fields': my_dict['f'], 'T': my_dict['T'][0][0], 'sx': my_dict['sx'][0][0]}

# If this file is run as a standalone, we should do benchmarking.
# Simulation parameters
n = 256
precision = jnp.float64

# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

# Construct a dictionary for grid information
param_dict = mhd_jax.construct_domain(n, precision)

# Get grids
x = param_dict['x']
y = param_dict['y']

segments = input_dict['fields'].shape[0]
nu = 1/100
eta = 1/100
b0 = [0.0, 0.1]  # Mean magnetic field
forcing = -4*jnp.cos(4*y)

# Append the extra system information to param_dict
param_dict.update( {'nu': nu, 'eta': eta, 'b0': b0, 'forcing': forcing} )

#for 32 segments of 8 steps each
param_dict.update( {'steps': 8} )







obj = lambda input_dict: mhd_jax.objective_RPO(input_dict, param_dict)
jit_obj = jax.jit(obj)

f = jit_obj(input_dict)

print(f.shape)

print(f"mean(abs(f)) = {jnp.mean( jnp.abs(f) )}")




vec_to_dict = lambda v: {'fields': jnp.reshape(v[0:-2], [segments, 2, n, n]), 'T': v[-2], 'sx': v[-1] }

jac = lambda v: jax.jvp( obj, primals=(input_dict,), tangents=(vec_to_dict(v),) )[1] # add [1] to only get the Jacobian action
jit_jac = jax.jit(jac)

'''
#print("Input dict:", input_dict)
for key, value in input_dict.items():
    print(f"Shape of primal {key}: {value.shape}")

tangent_dict = vec_to_dict(f)
#print("Tangent dict:", tangent_dict)
for key, value in tangent_dict.items():
    print(f"Shape of tangent {key}: {value.shape}")
'''

Jf = jit_jac(f)
print(f"mean(abs(Jf)) = {jnp.mean( jnp.abs(Jf) )}")

#Start Newton-Raphson iteration
def newton_step( input_dict ):
    f = jit_obj( input_dict )
    inner = 4
    damp = 1e-2
    ds, _ = jax.scipy.sparse.linalg.gmres( jit_jac, f, maxiter=inner )
    input_dict.update( { 'fields': input_dict['fields'] - damp*jnp.reshape(ds[0:-2], input_dict['fields'].shape) , 'T': input_dict['T'] - damp*ds[-2], 'sx': input_dict['sx'] - damp*ds[-1] } )
    return input_dict, f
jit_newton_step = jax.jit(newton_step)






newton_iterations = 512
for it in range(newton_iterations):
    start = time.time()
    input_dict, f = jit_newton_step(input_dict)
    stop = time.time()

    print(f"{it}: |f| = {jnp.linalg.norm(f)}, walltime = {stop - start}, T = { input_dict['T'] * segments}")

    if(it % 32 == 0):
        savemat(f"newton_output_{it}.mat", input_dict)