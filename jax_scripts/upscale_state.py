'''
The purpose of this script is to take an existing state and painlessly change the resolution.
'''
import jax
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO

#Filenames for input/output
infile = "solutions/Re200/1.npz"
outfile = "high_res.npz"

input_dict, param_dict = dictionaryIO.load_dicts(infile)
print(f"Loaded state... Current resolution of x is {param_dict['x'].shape}.")

#Define the new target resolution
n = 512
print(f"Attempting a change of resolution to {n}")


jax.config.update("jax_enable_x64", True)






def change_resolution( f, n ):
    m = f.shape[-1] #current resolution

    f = jnp.fft.fft2(f)
    f2 = jnp.zeros( [n,n], dtype=jnp.complex128)
   
    if m < n:
        #We are upsampling
        m2 = m//2

        #Fill in the four corners
        #I think this code ignores the Nyquist frequency, which is fine.
        f2 = f2.at[0:m2, 0:m2].set(  f[0:m2,0:m2] )
        f2 = f2.at[n-m2:,0:m2].set(  f[-m2:,0:m2] )
        f2 = f2.at[0:m2, n-m2:].set( f[0:m2,-m2:] )
        f2 = f2.at[n-m2:,n-m2:].set( f[-m2:,-m2:] )
    else:
        print("Error: downsampling not supported...")
        exit()
    f2 = jnp.fft.ifft2(f2) * (n/m)**2
    print(f2.shape)
    return jnp.real(f2)



#Construct a new grid
param_dict.update( mhd_jax.construct_domain( n, data_type=jnp.float64 ) )

#Change resolution of fields
print(f"mean squared fields = {jnp.mean(jnp.square(input_dict['fields'])):.6e} before resolution change")
change_res2 = jax.vmap( lambda f: change_resolution(f,n))
input_dict["fields"] = change_res2(input_dict["fields"])
print(f"mean squared fields = {jnp.mean(jnp.square(input_dict['fields'])):.6e} after resolution change")

#Also change the forcing
print(f"mean squared fields = {jnp.mean(jnp.square(param_dict['forcing'])):.6e} before resolution change")
param_dict["forcing"] = change_resolution(param_dict["forcing"], n)
print(f"mean squared fields = {jnp.mean(jnp.square(param_dict['forcing'])):.6e} after resolution change")

dictionaryIO.save_dicts(outfile, input_dict, param_dict)