'''
Let's debug a function for jax to do adjoint descent without blowing up our memory requirements
'''

import time
import jax
import jax.flatten_util
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
import lib.adam as adam
from lib.linalg import gmres, block_gmres


from scipy.io import savemat, loadmat

###############################
# Construct numerical grid
###############################

n = 128 # grid resolution
precision = jnp.float64  # Double or single precision

# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

# Generate grid information
param_dict = mhd_jax.construct_domain(n, precision)

#Pull out grid matrices for forcing construction
x = param_dict['x']
y = param_dict['y']




#################################################
# Physical parameters: dissipation, forcing, ...
#################################################
nu  = 1/40  # hydro dissipation
eta = 1/40  # magnetic dissipation

# Mean magnetic field
b0 = [0.0, 0.1]

# Construct your forcing
forcing = -4*jnp.cos(4*y)

param_dict.update({'forcing': forcing, 'b0': b0, 'nu': nu, 'eta': eta})


#Load the turbulent trajectory
data = jnp.load("turb.npz")
fs = data['fs']

#MATLAB indices I picked from visually inspecting the recurrence diagram.
idx = [147, 189]
f = fs[idx[0], :, :, :]
f = jnp.fft.irfft2(f)

dt = 0.005
ministeps = 32
T = dt * ministeps * (idx[1] - idx[0])
sx = 0.0





###########################################
# Load an initial condition from turbulence
###########################################

#Create a dictionary of optimizable field
input_dict = {"fields": f, "T": T, "sx": sx}

#Add the number of steps we need
param_dict.update({ 'steps': ministeps * (idx[1] - idx[0]) } )




#load a previous guess
matlab_data = loadmat("data/RPO_candidate_72.mat")
input_dict = {"fields": matlab_data['fields'], "T": matlab_data['T'][0][0], "sx": matlab_data['sx'][0][0] }
param_dict['steps'] = 1600


################################
# Test Jacobian Vector Products
################################

objective = jax.jit( lambda input_dict: loss_functions.objective_RPO(input_dict, param_dict) )

#Compile it
_ = objective(input_dict)

start = time.time()
f = objective(input_dict)
stop = time.time()

print( f"walltime = {stop - start} to evaluate the objective function")





jac = jax.jit( lambda primal, tangent: jax.jvp( objective, (primal,), (tangent,))[1] )

#compile
_ = jac( input_dict, f )

#Add phase conditions
jac_with_phase = jax.jit( lambda primal, tangent: loss_functions.add_phase_conditions( primal, tangent, jac(primal, tangent), param_dict) )

_ = jac_with_phase( input_dict, f )

start = time.time()
df = jac_with_phase( input_dict, f )
stop = time.time()

print( f"walltime = {stop - start} to evaluate the JVP with phase conditions")



#Try batched JVP for construction of a batched Krylov subspace. Freeze the input dict and vmap over the range
f_vec, unravel = jax.flatten_util.ravel_pytree(f)
f1 = lambda tangents: jac_with_phase(input_dict, tangents)
f2 = lambda x: jax.flatten_util.ravel_pytree(x)[0]

#lambda that acts on a vector and returns a vector. We handle the dict <-> vector translation
vector_jac = lambda vec: f2(f1(unravel(vec)))

batched_jac = jax.vmap( vector_jac, in_axes=1, out_axes=1 )

'''
for batch_size in range(16):    
    key = jax.random.PRNGKey(0)
    many_vectors = jax.random.normal( key, shape=(f_vec.shape[0], batch_size ), dtype=jnp.float64 )

    start = time.time()
    output = batched_jac( many_vectors )
    stop = time.time()
    print( f"{batch_size}, {stop - start}" )
'''

batch_size = 8
B = jnp.zeros((f_vec.shape[0], batch_size))

#Let's generate a realistic subspace
fi = jnp.zeros( (2,n,n), dtype=precision )
for i in range(batch_size):
    fi = fi.at[0,:,:].set( jnp.cos( (i+1)*x ) )
    B = B.at[:,i].set( f2( {'fields': fi, 'T': 0, 'sx': 0} ) )

m = 5
start = time.time()
block_gmres( batched_jac, f_vec, m, B, tol=1e-8)
stop = time.time()
print(f"walltime {stop-start}")