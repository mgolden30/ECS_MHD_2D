'''
Functions for integrating the 2D MHD equations in JAX. This solver is fully differentiable.

I've learned that using dicts as inputs to functions is much more pleasant for coding and debugging. Each function should take ~3 arguments max. Use lambdas liberally.

This library should contain exclusively functions for forward time integration. 
NO MACHINE LEARNING METHODS IN THIS LIBRARY.
'''



import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

#Tell the user if they are using the GPU or not when they invoke this library.
print(f"Jax is using {jax.devices()}\n")


def construct_domain(n: int, data_type):
    '''
    PURPOSE:
    This function creates a dictionary containing all static arrays that we need for
    2D spectral methods.

    INPUT:
    n - number of grid points per side
    data_type - jnp.float32 or jnp.float64

    OUTPUT:
    param_dict - a dictionary of useful grid info
       'x': x coordinate 
       'y': y coordinate
       'kx': wavenumbers in the x direction. rfft2 convention makes size [n,1]
       'ky': wavenumbers in the y direction. rfft2 convention makes size [1,n//2+1]
       'mask': 2/3rds dealiasing mask. size [n,n//2+1]
       'to_u': uncurling matrix  ky/(kx^2 + ky^2). size [n,n//2+1]
       'to_v': uncurling matrix -kx/(kx^2 + ky^2). size [n,n//2+1]
    '''

    # You should be requesting single of double precision
    assert ((data_type == jnp.float32) or (data_type == jnp.float64))

    # Build a 1D coordinate vector of periodic coordinates from [0, 2*pi)
    # I don't trust jnp.pi to be accurate to beyond single precision,
    # so just hard code in the value of pi
    grid = jnp.arange(n, dtype=data_type) / n * 2 * 3.14159265358979323846

    # Turn these into [n,n] coordinate matrices.
    # x will vary along the first dimension and y along the second.
    # (anti-MATLAB indexing)
    x, y = jnp.meshgrid(grid, grid, indexing='ij')

    # Integer wavenumbers 0,1,...n/2, -n/2+1, ..., -1
    k = jnp.fft.fftfreq(n, d=1/n, dtype=data_type)

    if data_type == jnp.float64:
        assert (jnp.abs(k[1] - 1) < 1e-12)

    # The output of a real fourier transform will be [n,n//2+1]
    kx = jnp.reshape(k,          [-1, 1])
    ky = jnp.reshape(k[:n//2+1], [1, -1])
    
    #Two thirds dealiasing
    mask = (jnp.abs(kx) < n/3) & (jnp.abs(ky) < n/3)
    
    #Also mask out the zero mode
    #JAX complained if I tried to set the value to 0 since mask is a boolean array. 
    #Set to false to remove the error.
    mask = mask.at[0, 0].set(False)

    k_sq = kx*kx + ky*ky
    # Prevent division by zero
    k_sq = k_sq.at[0, 0].set(1)


    #inverse_laplacian
    inv_lap = -1/k_sq
    inv_lap = inv_lap.at[0,0].set(0)
    inv_lap = inv_lap*mask

    # uncurl matrices
    to_u =  ky / k_sq
    to_v = -kx / k_sq

    # Pre-mask derivatives and uncurling
    kx = kx*mask
    ky = ky*mask
    to_u = to_u*mask
    to_v = to_v*mask

    param_dict = {'x': x, 'y': y, 'kx': kx, 'ky': ky, 'mask': mask, 'to_u': to_u, 'to_v': to_v, 'inv_lap': inv_lap}

    return param_dict


def dissipation(param_dict):
    #Construct a dissipation operator
    k_sq = param_dict['kx']**2 + param_dict['ky']**2
    coeffs = jnp.array([param_dict['nu'], param_dict['eta']])
    coeffs = jnp.reshape(coeffs, [2,1,1])
    return - k_sq * coeffs


def state_vel(fields, param_dict, include_dissipation ):
    '''
    PURPOSE:
    Compute the state velocity of the pair (w,j) where w is vorticity and j is current.
    
    INPUT:
    fields - tensor of shape [2, n, n//2+1]. Assume this is the output of jnp.fft.rfft2.
             fields[0,:,:] - vorticity w coefficients
             fields[1,:,:] - current j coefficients
             
    param_dict - an output of construct_domain
    
    include_dissipation - a boolean flag to indicate if the dissipation 
                          should be accounted for. For semi-implicit integration,
                          we account for the dissipation separately to avoid small 
                          timesteps. However for Newton-Raphson iteration, we would
                          to compute the whole derivative and include dissipation.
    '''

    #Pull derivative matrices from dict
    kx = param_dict['kx']
    ky = param_dict['ky']
    mask = param_dict['mask']
    to_u = param_dict['to_u']
    to_v = param_dict['to_v']
    
    #Extra parameters we also pull from the param_dict
    forcing = param_dict['forcing']
    b0 = param_dict['b0']

    # Spatial derivative
    fx = jnp.fft.irfft2(1j * kx * fields)
    fy = jnp.fft.irfft2(1j * ky * fields)

    # uncurled vector components
    fu = jnp.fft.irfft2(1j * to_u * fields)
    fv = jnp.fft.irfft2(1j * to_v * fields)

    #Define the mean field components to broadcast
    #I should also add in mean flow
    bx = jnp.reshape( jnp.array( [0, b0[0]] ), [2,1,1] )
    by = jnp.reshape( jnp.array( [0, b0[1]] ), [2,1,1] )
    
    fu = fu + bx
    fv = fv + by

    #Note this computes the u dot grad w and B dot grad j
    advection = fu * fx + fv * fy

    k_sq = (kx*kx + ky*ky)*mask

    # vorticity dynamics
    #weight = jnp.reshape( jnp.array([-1,1]), [2,1,1] )
    #dwdt = jnp.sum( advection*weight, axis=-3 ) + forcing
    dwdt = -advection[..., 0, :, :] + advection[..., 1, :, :] + forcing
    dwdt = mask * jnp.fft.rfft2(dwdt)

    # Current dynamics
    djdt = fu[..., 0, :, :]*fv[..., 1, :, :] - fv[..., 0, :, :]*fu[..., 1, :, :]
    djdt = k_sq * jnp.fft.rfft2(djdt)
        
    # Get a total state velocity
    dwdt = jnp.expand_dims(dwdt, axis=-3)
    djdt = jnp.expand_dims(djdt, axis=-3)
    dfdt = jnp.concatenate([dwdt, djdt], axis=-3)

    if include_dissipation:
        diss = dissipation(param_dict)
        dfdt += diss * fields

    return dfdt

def vis(f):    
    figure, axis = plt.subplots(1,2)

    f = jnp.fft.irfft2(f)

    axis[0].imshow( f[0,:,:].transpose(), vmin=-10, vmax=10, cmap='bwr', origin='lower')
    axis[0].set_title("vorticity")
    
    axis[1].imshow( f[1,:,:].transpose(), vmin=-10, vmax=10, cmap='bwr', origin='lower' )
    axis[1].set_title("current")
    return figure, axis
















if (__name__ == "__main__"):
    print( f"Running {__file__} as main.")

    import time
    import timestepping

    # If this file is run as a standalone, we should do benchmarking.
    n = 256
    precision = jnp.float64
    if (precision == jnp.float64):
        jax.config.update("jax_enable_x64", True)

    # Construct a dictionary for grid information
    param_dict = construct_domain(n, precision)

    # Get grids
    x = param_dict['x']
    y = param_dict['y']

    nu  = 1/50
    eta = 1/50
    b0  = [0.0, 0.1]  # Mean magnetic field
    forcing = -4*jnp.cos(4*y)

    # Append the extra system information to param_dict
    param_dict.update( {'nu': nu, 'eta': eta, 'b0': b0, 'forcing': forcing} )

    # Initial data
    f = jnp.zeros([2, n, n], dtype=precision)
    f = f.at[0, :, :].set(jnp.cos(x-0.1)*jnp.sin(x+y-1.2) + jnp.sin(3*x-1)*jnp.cos(y-1))
    f = f.at[1, :, :].set(jnp.cos(x+2.1)*jnp.sin(y+3.5))
    f = 10*f

    #fft the data before we evolve
    f = jnp.fft.rfft2(f)

    #Do some benchmarking of Jit
    def jit_test( fn, trials ):
        print(f"Running {trials} trials.")
        start = time.time()
        for _ in range(trials):
            _ = fn()
        stop = time.time()
        no_jit_time = stop-start
        print(f"{no_jit_time:.3e} seconds (no jit)")

        jit_fn = jax.jit(fn)
        _ = jit_fn() #compile on first function call

        start = time.time()
        for _ in range(trials):
            _ = jit_fn()
        stop = time.time()
        with_jit_time = stop-start
        print(f"{with_jit_time:.3e} seconds (jit)")
        print(f"jit provided a x{no_jit_time/with_jit_time:.2f} speedup\n")
        print("")

    trials = 128
    v_fn = lambda : state_vel(f, param_dict, include_dissipation=True)
    print("Benchmarking velocity...")
    jit_test( v_fn, trials )

    #redefine v_fn to take an argument
    v_fn = lambda f : state_vel(f, param_dict, include_dissipation=True)  

    t = 1.00
    steps = 256
    fn = lambda : timestepping.rk4(f, t, steps, v_fn)
    print(f"Benchmarking DNS with RK4...")
    trials = 4
    jit_test( fn, trials )


    #Let's also do a quick convergence test
    steps_fine = 4*1024
    f_fine = timestepping.rk4(f, t, steps_fine, v_fn)

    steps_array = jnp.array( [ 512, 800, 1024])
    err = jnp.zeros((len(steps_array),))
    for i in range(len(steps_array)):
        steps = steps_array[i]
        f_out = timestepping.rk4(f, t, steps, v_fn)

        err =  err.at[i].set(jnp.sqrt(jnp.mean( jnp.square(jnp.fft.irfft2(f_out - f_fine)))))
    print(f"errors are {err}")

    poly = jnp.polyfit( jnp.log(steps_array), jnp.log(err), deg=1 )
    print(f"Empirical power law fit gives an exponent of {poly[0]:.3f}")
