'''
Written June 19 (Happy Juneteenth) 2025 by Matthew Golden

PURPOSE:
Let's converge equilibria + traveling waves for 2D MHD.

VERSION HISTORY

'''

import jax
import jax.flatten_util
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
import lib.adam as adam
from lib.newton_krylov import newton_krylov

from scipy.io import savemat, loadmat





########################################
# Simulation parameters
########################################

n = 1024  # grid resolution
precision = jnp.float64 # Double or single precision

# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

# Generate grid information
param_dict = mhd_jax.construct_domain(n, precision)


nu  = 1/100
eta = 1/100
b0  = [0.0, 1.0] # Mean magnetic field

x = param_dict['x']
y = param_dict['y']

# Construct your forcing
forcing = -4*jnp.cos(4*y)

param_dict.update({'forcing': forcing, 'b0': b0, 'nu': nu, 'eta': eta})






###############################
# Initial data
###############################

#f will contain fluid and magnetic field info
f = jnp.zeros([2, n, n], dtype=precision)

# EQ1
#f = f.at[0,:,:].set( jnp.cos(x)*jnp.cos(y) + jnp.sin(3*x-1)*jnp.cos(y-1) )

# EQ2
#f = f.at[0,:,:].set( jnp.cos(x) + jnp.cos(y) )

# EQ3
#f = f.at[1,:,:].set( jnp.cos(x) + jnp.cos(y) )


# EQ4
#f = f.at[0,:,:].set( jnp.cos(x+2*y-1) + jnp.cos(x-y) )
#f = f.at[1,:,:].set( jnp.cos(x) + jnp.cos(y + 1) )

#Playing around
#f = f.at[1,:,:].set( jnp.cos(x) + jnp.cos(y) )
f = f.at[0,:,:].set( 10*jnp.sin(x) )
#f = f.at[1,:,:].set( jnp.cos(x) - jnp.cos(y) )
wave_speed = 0.0


#Load an initial condition
data = loadmat("dist2.mat")
f = f.at[:,:,:].set( data['fs'][0,:,:,:] )
input_dict = {'fields': f, 'wave_speed': wave_speed}



##################
# Optimization 
##################

#mode = "adam"
mode = "newton"

if mode == "adam":
    data = loadmat("solutions/Re100/EQ7.mat")
    input_dict["fields"] =  jnp.array( data['f'], dtype=precision )
    input_dict["wave_speed"] = jnp.array( data['c'], dtype=precision )
    maxit = 1024*8*5

    loss_fn = jax.jit(loss_functions.traveling_wave_loss)
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

    m, v = adam.init_adam(input_dict)
    update_fn = jax.jit(adam.adam_update)

    for t in range(maxit):
        loss = loss_fn( input_dict, param_dict )
        grad = grad_fn( input_dict, param_dict)

        # ADAM optimization
        input_dict, m, v = update_fn( input_dict, grad, m, v, t+1, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-6)

        #Dealias
        f = input_dict['fields']
        f = jnp.fft.rfft2(f) * param_dict['mask']
        f = jnp.fft.irfft2(f)
        input_dict['fields'] = f

        if (t % 128 == 0):
            print(f"{t}: loss = {loss}")

    savemat("solutions/Re100/EQ7.mat", {"f": input_dict['fields'], "c": input_dict['wave_speed'], "B0": b0, "eta": eta, "nu": nu })
    exit()


if mode == "newton":
    data = loadmat("solutions/Re100/EQ7.mat")
    input_dict["fields"] =  jnp.array( data['f'], dtype=precision )
    input_dict["wave_speed"] = jnp.array( data['c'], dtype=precision )

    #Function for root finding
    F_dict = lambda input_dict: loss_functions.traveling_wave_objective(input_dict, param_dict)


    def M( x, mode):
        '''
        Precondition our linear system
        '''

        #Self adjoint, so no mode handling needed
        _ = mode

        f = jnp.reshape(x, [2,n,n])
        f = jnp.fft.rfft2(f)

        #Swap u and B
        #f = jnp.roll(f, shift=1, axis=0)

        #Apply an inverse laplacian
        #f = f*jnp.sqrt(-param_dict['inv_lap'])
        #f = f*param_dict['inv_lap']

        #f = f*jnp.sqrt( param_dict['kx']**2 + param_dict['ky']**2 )
        f = f*param_dict['mask']

        #Undo Fourier transform
        f = jnp.fft.irfft2(f)

        #Return column vector
        return jax.flatten_util.ravel_pytree(f)[0]


    def dealias(x):
        f = x['fields']
        f = jnp.fft.rfft2(f)
        f = f*param_dict['mask']
        f = jnp.fft.irfft2(f)
        x['fields'] = f   
        return x

    maxit = 32
    inner = 128
    damp  = 1.0
    input_dict = newton_krylov( F_dict, input_dict, maxit, inner, damp, M, dealias )
    savemat("solutions/Re100/EQ7.mat", {"f": input_dict['fields'], "c": input_dict['wave_speed'], "B0": b0, "eta": eta, "nu": nu })