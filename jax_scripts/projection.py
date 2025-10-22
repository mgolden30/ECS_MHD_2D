import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import imageio.v2 as imageio

import lib.mhd_jax as mhd_jax
import lib.timestepping as timestepping
import lib.dictionaryIO as dictionaryIO

from scipy.io import savemat
from pathlib import Path

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

def compute_observables( f, param_dict ):
    '''
    Compute the energy injection and dissipation

    Parameters
    ----------
    f : ndarray
        shape [2,n,n//2+1] field. u and B in Fourier space
    param_dict : dictionary
    '''
    
    
    #uncurl the forcing and the vorticity
    u = jnp.fft.irfft2(1j * param_dict['to_u'] * f[0,:,:])
    v = jnp.fft.irfft2(1j * param_dict['to_v'] * f[0,:,:])
    
    force = jnp.fft.rfft2(param_dict['forcing'])
    fx = jnp.fft.irfft2(1j * param_dict['to_u'] * force)
    fy = jnp.fft.irfft2(1j * param_dict['to_v'] * force)

    #Compute the energy injection
    injection = jnp.mean( u*fx + v*fy )

    #Dissipation is easy to compute
    dissipation = jnp.mean( jnp.square(jnp.fft.irfft2(f)), axis=[-1,-2] )
    dissipation = param_dict['nu'] * dissipation[0] + param_dict['eta'] * dissipation[1]
    return jnp.array([injection, dissipation])



def laminar_flow(param_dict):
    '''
    Construct the laminar flow

    Assuming forcing = sin(4y) in x direction

    '''

    alpha = param_dict['b0'][1] #y component of the magnetic field

    L = jnp.sqrt( param_dict['eta'] * param_dict['nu'] ) / alpha

    n = param_dict['x'].shape[0]
    f = jnp.zeros((2,n,n))

    k = 4.0
    y = param_dict['y']

    U0 = param_dict['eta'] / (alpha*alpha * (1 + L*L*k*k))
    B0 = 1.0 / (k * alpha * (1 + L*L*k*k))

    f = f.at[0,:,:].set(  -U0*k*jnp.cos(k*y) )
    f = f.at[1,:,:].set(  B0*k*jnp.sin(k*y) )
    return f

directory = Path("./solutions/Re40")

#empty dictionary
projection = {}


for file in directory.rglob("*.npz"):
    print(file)
    input_dict, param_dict = dictionaryIO.load_dicts(file)

    param_dict = dictionaryIO.recompute_grid_information(input_dict, param_dict)
    
    
    
    #Move to Fourier space
    f = jnp.fft.rfft2(input_dict['fields'])

    #allocate memory for the projection
    obs = jnp.zeros((2,param_dict['steps']))

    dt = input_dict['T'] / param_dict['steps']
    v_fn = lambda f: mhd_jax.state_vel(f, param_dict, include_dissipation=True)
    one_step = jax.jit(lambda f : timestepping.tdrk4(f, dt, 1, v_fn))

    def loop_body(i, carry):
        obs, f = carry
        obs = obs.at[:,i].set(compute_observables(f, param_dict))
        f = one_step(f)
        return (obs, f)
    
    obs, f = jax.lax.fori_loop(0, param_dict['steps'], loop_body, (obs,f))




    f_lam = laminar_flow(param_dict)
    f_lam = jnp.fft.rfft2(f_lam)
    v_lam = jnp.fft.irfft2(v_fn(f_lam))
    print(f"max velocity of laminar is {jnp.max(jnp.abs(v_lam))}")

    #normalize by laminar
    obs = obs / jnp.reshape( compute_observables(f_lam, param_dict), [2,1] )

    projection.update({"proj" + file.stem: obs})

mat_path = "proj.mat"
savemat(mat_path, projection)