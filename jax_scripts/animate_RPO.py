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

# By = 1.0, Re = 100
input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE5.npz")
input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_336.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("newton/3.npz")

input_dict, param_dict = dictionaryIO.load_dicts("data/adjoint_descent_12608.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("high_res.npz")

f = input_dict['fields']
T = input_dict['T']
sx= input_dict['sx']

steps = param_dict['steps']
dt = T/steps


evolve = jax.jit( lambda f: mhd_jax.eark4(f, dt, ministeps, param_dict) )


if f.ndim == 4:
    mode = "multi_shooting"
else:
    mode = "single_shooting"

#ministeps = integration steps to take between frames
ministeps = 8
macrosteps = steps // ministeps
print(f"Saving {macrosteps} = {steps}//{ministeps} frames...")

if mode == "single_shooting":
    savemat(f"traj/0.mat", {"f":f, "sx": sx})
    for t in range( macrosteps ):
        print(t)
        f = jnp.fft.rfft2(f)
        f = evolve(f)
        f = jnp.fft.irfft2(f)
        savemat(f"traj/{t}.mat", {"f":f, "sx": sx})

if mode == "multi_shooting":
    print(f"param_dict['ministeps'] = {param_dict['ministeps']}")
    print(f"param_dict['ministeps']/ministeps = {param_dict['ministeps']/ministeps}")
    
    #Assume they divide nicely
    assert( param_dict['ministeps'] % ministeps == 0 )

    restart = param_dict['ministeps'] // ministeps

    #Loop over this
    for i in range(macrosteps):
        print(i)
        if i % restart == 0:
            print("Switching to new initial condition...")
            g = f[i//restart, ...]
        savemat(f"traj/{i}.mat", {"f":g, "sx": sx})
        g = jnp.fft.rfft2(g)
        g = evolve(g)
        g = jnp.fft.irfft2(g)
