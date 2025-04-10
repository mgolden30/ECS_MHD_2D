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

#print( df )



import numpy as np

def save_jax_dict(filename, jax_dict):
    flat, treedef = jax.tree_util.tree_flatten(jax_dict)
    np_flat = [np.array(x) for x in flat]  # convert to numpy
    np.savez(filename, *np_flat, treedef=treedef)

def load_jax_dict(filename):
    data = np.load(filename, allow_pickle=True)
    treedef = data['treedef'].item()
    flat = [data[key] for key in data.files if key != 'treedef']
    return jax.tree_util.tree_unflatten(treedef, flat)

#Attempt Newton-Raphson iteration. God be kind, forgive me for the sins I am about to commit.


input_dict = load_jax_dict("newton/best.npz")
'''

macrosteps = param_dict['steps'] // ministeps
        
f = input_dict["fields"]
T = input_dict['T']
dt= T/param_dict['steps']

update = jax.jit( lambda f: mhd_jax.eark4(f, dt, ministeps, param_dict) )

f = jnp.fft.rfft2(f)
savemat( f"timeseries/0.mat", {"f": jnp.fft.irfft2(f), "T": T, "sx": input_dict["sx"] } )
for i in range(macrosteps):
    f = update(f)
    savemat( f"timeseries/{i+1}.mat", {"f": jnp.fft.irfft2(f), "T": T, "sx": input_dict["sx"] } )
exit()
'''

maxit = 1
inner = 8
outer = 1
damp  = 1.0

 


#Compile an adjoint looping function that picks out a descent direction to kick off Newton-GMRES
segments = 4 #Break up integration to reduce memory requirement
grad_fn  = lambda input_dict: loss_functions.loss_RPO_memory_efficient( input_dict, param_dict, segments )[1]
grad_fn = jax.jit(grad_fn)


#Compile
_ = grad_fn(input_dict)

for i in range(maxit):
    start = time.time()
    f = objective(input_dict)
    stop = time.time()
    fwalltime = stop - start

    f_vec, unravel_fn = jax.flatten_util.ravel_pytree(f)

    #Define a linear operator for GMRES
    #Do this every iteration since input_dict changes
    lin_op = lambda v:  jax.flatten_util.ravel_pytree(  jac_with_phase( input_dict, unravel_fn(v))  )[0]

    #Solve for a Newton step with GMRES
    start = time.time()
    
    #Get initial guess for GMRES from adjoint descent
    Q0 = jax.flatten_util.ravel_pytree( grad_fn(input_dict ) )[0]

    #step = gmres(lin_op, f_vec, inner, Q0)
    stop = time.time()
    gmreswalltime = stop - start

    f1 = lambda tangents: jac_with_phase(input_dict, tangents)
    f2 = lambda x: jax.flatten_util.ravel_pytree(x)[0]

    #lambda that acts on a vector and returns a vector. We handle the dict <-> vector translation
    vector_jac = lambda vec: f2(f1(unravel_fn(vec)))
    batched_jac = jax.vmap( vector_jac, in_axes=1, out_axes=1 )

    batch_size = 8
    B = jnp.zeros((f_vec.shape[0], batch_size))

    #Let's generate a realistic subspace
    fi = jnp.zeros( (2,n,n), dtype=precision )
    B = B.at[:,0].set( f_vec )
    B = B.at[:,1].set( f2({'fields': jnp.zeros((2,n,n), dtype=precision), 'T': 1, 'sx':0}) )
    B = B.at[:,2].set( f2({'fields': jnp.zeros((2,n,n), dtype=precision), 'T': 0, 'sx':1}) )
    
    for ii in range(batch_size-3):
        fi = fi.at[0,:,:].set( jnp.cos( (ii+1)*param_dict['x'] ) )
        B  = B.at[:,ii+3].set( f2( {'fields': fi, 'T': 0, 'sx': 0} ) )

    m = 32
    start = time.time()
    step = block_gmres( batched_jac, f_vec, m, B, tol=1e-8)
    stop = time.time()
    print(f"walltime {stop-start}")

    savemat("f_vec.mat", {"b": f_vec})



    print(f"Iteration {i}: |f| = {jnp.mean(jnp.square(f_vec))}, fwalltime = {fwalltime}, gmres time = {gmreswalltime}")
    
    #update the input_dict
    x, unravel_fn = jax.flatten_util.ravel_pytree( input_dict )
    x = x - damp * step
    input_dict = unravel_fn(x)

    save_jax_dict(f"newton/{i}.bin", input_dict)