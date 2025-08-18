'''
Perform Newton-GMRES to converge RPOs
'''

import jax
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO

from scipy.io import savemat

###############################
# Construct numerical grid
###############################


precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)


input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_64.npz")
input_dict, param_dict = dictionaryIO.load_dicts("Re40/RPO1.npz")

#input_dict, param_dict = dictionaryIO.load_dicts("newton/5.npz")


ministeps = 16
steps = param_dict['steps']
assert( steps % ministeps == 0 )

f = input_dict['fields']
f = jnp.fft.rfft2(f)


dt = input_dict['T'] / steps

for i in range( steps // ministeps ):
    print(i)
    savemat(f"timeseries/{i:03d}.mat", {"f": jnp.fft.irfft2(f), "T": input_dict["T"], "sx": input_dict["sx"] })
    f = mhd_jax.eark4( f, dt, ministeps, param_dict)

