'''
Let's use JAX to hunt for RPOs
'''

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO
import time
import jax
import jax.numpy as jnp

from scipy.io import savemat, loadmat

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_648.npz")
#Kolmogorov Re=60
input_dict, param_dict = dictionaryIO.load_dicts("newton/12.npz")

# By = 1.0, Re = 100
input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE2.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE2.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_760.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("newton/15.npz")


for key, value in input_dict.items():
    print(f"Key: {key}, Value: not printing")

for key, value in param_dict.items():
    print(f"Key: {key}, Value: not printing")

f = input_dict['fields']
T = input_dict['T']
sx= input_dict['sx']

steps = param_dict['steps']
dt = T/steps

ministeps = 32
macrosteps = steps // ministeps

print(macrosteps)

evolve = jax.jit( lambda f: mhd_jax.eark4(f, dt, ministeps, param_dict) )

savemat(f"traj/0.mat", {"f":f, "sx": sx})
for t in range( macrosteps ):
    print(t)
    f = jnp.fft.rfft2(f)
    f = evolve(f)
    f = jnp.fft.irfft2(f)
    savemat(f"traj/{t}.mat", {"f":f, "sx": sx})