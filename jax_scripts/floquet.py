'''
Estimate the Floquet spectrum of a Relative Periodic Orbit (RPO)
'''
import time
import jax
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO

from scipy.io import savemat, loadmat

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re100/RPO_CLOSE2.npz") #_multi.npz")
#input_dict, param_dict = dictionaryIO.load_dicts("newton/2.npz")

f = input_dict['fields']

if f.ndim == 4:
    #Restrict to single shooting
    f = f[0,...]

@jax.jit
def forward(f):
    dt = input_dict['T'] / param_dict['steps']
    f = jnp.fft.rfft2(f)
    #evolve in time
    f = mhd_jax.eark4(f, dt, param_dict['steps'], param_dict)
    #Translate in space
    f = jnp.exp( -1j*param_dict['kx']*input_dict['sx'] ) * f
    f = jnp.fft.irfft2(f)
    return f

#Try using the evolution function.
f_out = forward(f)

norm = lambda x: jnp.linalg.norm( jnp.reshape(x, [-1]) )
relative_error = norm(f-f_out)/norm(f)

print(f"relative error in periodic orbit = {relative_error:.3e}")

jac =  lambda tangent: jax.jvp( forward, (f,), (tangent,))[1]
jac = jax.jit(jax.vmap(jac))

#How many tangent vectors do we want to iterate?
block_size = 32
n = f.shape[-1]
key = jax.random.PRNGKey(seed=0)
tang = jax.random.normal( key, [block_size, 2, n, n], dtype=precision )

#Or load in a previous convergence
data = loadmat("floquet.mat")
tang = data["tang"]

#How many times should we do power iteration?
maxit = 16
for i in range(maxit):
    print(tang.shape)

    start = time.time()
    tang = jac(tang)

    print( f"iteration {i}: walltime = {time.time() - start:.3f}" )

    #Orthonormalize
    tang = jnp.reshape(tang, [block_size, -1])
    tang, _ = jnp.linalg.qr(tang.transpose(), mode="reduced")
    tang = tang.transpose()
    tang = jnp.reshape(tang, [block_size, 2, n, n])

#After some estimation of the eigenvectors, apply it one more time to estimate the Schur factor
t  = jnp.reshape(     tang,  [block_size, -1] )
jt = jnp.reshape( jac(tang), [block_size, -1] )

R = t @ jt.transpose()

savemat("floquet.mat", {"R": R, "tang": tang, "diff": f_out - f })
