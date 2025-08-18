'''
Let's use JAX to hunt for RPOs
'''

import mhd_jax
import time
import jax
import jax.numpy as jnp

from scipy.io import savemat, loadmat

# Simulation parameters
n = 256  # grid resolution
precision = jnp.float64  # Double or single precision

# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

# Generate grid information
x, y, kx, ky, mask, to_u, to_v = mhd_jax.construct_domain(n, precision)

nu = 1/100  # hydro dissipation
eta = 1/100  # magnetic dissipation

# Mean magnetic field
b0 = [0.0, 0.1]

# Construct your forcing
forcing = -4*jnp.cos(4*y)

# my_dict = loadmat("converged.mat")
#my_dict = loadmat("jax_rk4.mat")
#f = f.at[:, :, :].set(my_dict["f"])

# Period
T  = 2.0
sx = 0.0
steps = 512

my_dict = loadmat("data/converged_RPO_4225.mat")
f_all = my_dict["f"]
#f  = f.at[:, :, :].set(my_dict["f"])
T  = my_dict["T"][0][0]
sx = my_dict["sx"][0][0]

print(T)
print(sx)

steps = 8
h = T/steps

segments = 32

for seg in range(segments):
    f = f_all[seg,:,:,:]
    savemat(f"traj/{seg}_0.mat", {"f": f})
    for t in range(steps):
        print(t)
        f = jnp.fft.rfft2(f)
        f = mhd_jax.rk4_step(f, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, h)
        f = jnp.fft.irfft2(f)
        savemat(f"traj/{seg}_{t+1}.mat", {"f":f})