import jax
import jax.numpy as jnp

def shift_x(f, s, param_dict):
    return jnp.exp(1j*param_dict['kx']*s)*f

def shift_reflect(f, n, param_dict):
    #Shift in y half a wavelength of the forcing
    #n - number of shift reflects to apply
    n = int(n)
    s = (n*2*jnp.pi) / 8
    f = jnp.exp(1j*param_dict['ky']*s)*f
    if n % 2 == 1:
        #Reflect in x
        f = jnp.flip(f,axis=-2) # k = [0,1,2,-1] -> [-1,2,1,0], using n=4 as a minimal example
        f = jnp.roll(f,shift=1, axis=-2) # k = [-1,2,1,0] -> [0,-1,2,1]
        #Change the sign of both fields
        f = -f
    return f

def rot180(f):
    #Rotate 
    f = jnp.conj(f)
    sign = jnp.reshape( jnp.array([1.0,-1.0]), [2,1,1] )
    return sign*f


def find_optimal_symmetry(f1, f2, param_dict, nx):
    '''
    Given two snapshots, find the optimal symmetry g such that |f1 - g(f2)| is minimal.
    
    This code assumes a forcing of f = -4cos(4y) and a mean magnetic field B = <0,By>

    INPUT
    f1 - first snapshot
    f2 - second snapshot
    param_dict - dictionary with grid information
    nx - number of shifts in x we want to try
    '''
    #Should be (s, ny, rot?)
    best_g = (0.0, 0, False)
    best_err = 1e100
    for i in range(nx):
        s = (2*jnp.pi/nx)*i #shift in x
        for ny in range(8):
            for rot in (True, False):
                f = shift_x(f2,s,param_dict)
                f = shift_reflect(f,ny,param_dict)
                if rot:
                    f = rot180(f)
                err = jnp.mean(jnp.square(jnp.abs(f1 - f)))
                if err < best_err:
                    best_err = err
                    best_g = (s, ny, rot)
    print(f"Best symmetry found is {best_g}")
    return best_g

if __name__ == "__main__":
    #Run a quick check that these symmetries commute with DNS.
    import mhd_jax as mhd_jax
    import dictionaryIO as dictionaryIO

    ######################
    # DNS parameters
    ######################
    n  = 256    #grid resolution
    dt = 1/256  #size of timestep
    steps = 256 #number of timesteps
    precision = jnp.float64 #double or single precision

    
    nu  = 1/50 #Fluid dissipation
    eta = 1/50 #Magnetic dissipation
    b0  = [0.0, 0.1] # Mean magnetic field 

    # If you want double precision, change JAX defaults
    if (precision == jnp.float64):
        jax.config.update("jax_enable_x64", True)

    # Construct a dictionary for grid information
    param_dict = mhd_jax.construct_domain(n, precision)

    # Get grids
    x = param_dict['x']
    y = param_dict['y']

    forcing = -4*jnp.cos(4*y)

    #f[0,:,:] is the vorticity and f[1,:,:] is the current density.
    f = jnp.zeros([2, n, n], dtype=precision)
    f = f.at[0, :, :].set( jnp.cos(4*x-0.1)*jnp.sin(x+y-1.2) - jnp.sin(3*x-1)*jnp.cos(y-1) + 2*jnp.cos(2*x-1))
    f = f.at[1, :, :].set( jnp.cos(3*x+2.1)*jnp.sin(y+3.5) - jnp.cos(1-x) + jnp.sin(x + 5*y - 1 ) )

    # Append the extra system information to param_dict
    param_dict.update( {'nu': nu, 'eta': eta, 'b0': b0, 'forcing': forcing} )
    #Append timestepping information as well
    param_dict.update( {'dt': dt, 'steps': steps} )

    #fft the data before we evolve
    f = jnp.fft.rfft2(f)
    f = param_dict['mask'] * f

    evolve = jax.jit( lambda f: mhd_jax.eark4(f, dt, steps, param_dict) )
    _ = evolve(f)

    #################
    # shift in x
    #################
    s = 1.0
    symm = lambda f: shift_x(f, s, param_dict)
    diff = symm(evolve(f)) - evolve(symm(f))
    print(f"Shift in x has a max commutator error of {jnp.max(jnp.abs(diff)):.3e}")


    #####################
    # shift-reflect
    #####################
    for n in range(8):
        symm = lambda f: shift_reflect(f, n, param_dict)
        diff = symm(evolve(f)) - evolve(symm(f))
        print(f"Shift-reflect has a max commutator error of {jnp.max(jnp.abs(diff)):.3e}")

    #####################
    # rotation 180
    #####################
    symm = lambda f: rot180(f)
    diff = symm(evolve(f)) - evolve(symm(f))
    print(f"Shift-reflect has a max commutator error of {jnp.max(jnp.abs(diff)):.3e}")