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

    # Check that fftfreq doesn't fuck up the precision of k
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

    # uncurl matrices
    to_u =  ky / k_sq
    to_v = -kx / k_sq

    # Pre-mask derivatives and uncurling
    kx = kx*mask
    ky = ky*mask
    to_u = to_u*mask
    to_v = to_v*mask

    param_dict = {'x': x, 'y': y, 'kx': kx, 'ky': ky, 'mask': mask, 'to_u': to_u, 'to_v': to_v}

    return param_dict



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
    nu  = param_dict['nu']
    eta = param_dict['eta']
    forcing = param_dict['forcing']
    b0 = param_dict['b0']

    # Spatial derivative
    fx = jnp.fft.irfft2(1j * kx * fields)
    fy = jnp.fft.irfft2(1j * ky * fields)

    # uncurled vector components
    fu = jnp.fft.irfft2(1j * to_u * fields)
    fv = jnp.fft.irfft2(1j * to_v * fields)

    # Add mean magnetic field
    fu = fu.at[1, :, :].set(fu[1, :, :] + b0[0])
    fv = fv.at[1, :, :].set(fv[1, :, :] + b0[1])

    #Note this computes the u dot grad w and B dot grad j
    advection = fu * fx + fv * fy

    k_sq = kx*kx + ky*ky

    # vorticity dynamics
    dwdt = -advection[0, :, :] + advection[1, :, :] + forcing
    dwdt = mask * jnp.fft.rfft2(dwdt)

    # Current dynamics
    djdt = fu[0, :, :]*fv[1, :, :] - fv[0, :, :]*fu[1, :, :]
    djdt = k_sq * jnp.fft.rfft2(djdt)

    if include_dissipation:
        dwdt += - nu  * k_sq * fields[0,:,:]
        djdt += - eta * k_sq * fields[1,:,:]
        
    # Get a total velocity
    dwdt = jnp.expand_dims(dwdt, axis=0)
    djdt = jnp.expand_dims(djdt, axis=0)
    dfdt = jnp.concatenate([dwdt, djdt], axis=0)
    return dfdt


def eark4_step(f, dt, param_dict, diss):
    '''
    A step of exponential ansatz Runge-Kutta of 4th order (EARK4). 
    We do operator splitting to handle the dissipation implicitly and avoid small timesteps.
    '''
    
    #Don't compute dissipation explicitly
    vel = lambda f: dt * state_vel( f, param_dict, include_dissipation=False )

    #EARK4 looks like RK$ but with exponential twiddles that you interleave.
    k1 = vel(f)

    f  =  f*diss
    k1 = k1*diss
    
    k2 = vel(f + k1/2)
    k3 = vel(f + k2/2)
   
    f  =  f*diss
    k1 = k1*diss
    k2 = k2*diss
    k3 = k3*diss
   
    k4 = vel(f + k3)
    
    f = f + (k1 + 2*k2 + 2*k3 + k4)/6
    
    return f





def eark4(f, dt, steps, param_dict ):
    '''
    Perform many steps of Exponential Ansatz Runge-Kutta 4 (EARK4)
    '''

    #Construct a diagonal dissipation operator
    diss = jnp.zeros_like(f)
    k_sq = jnp.square(param_dict['kx']) + jnp.square(param_dict['ky'] )
    diss = diss.at[0,:,:].set( jnp.exp( -param_dict['nu']  * k_sq * dt / 2 ) )
    diss = diss.at[1,:,:].set( jnp.exp( -param_dict['eta'] * k_sq * dt / 2 ) )
    diss = diss * param_dict['mask']

    #Need this lambda format to use fori_loop
    #Might as well jit it since we call this update a lot
    update_f = jax.jit( lambda _, f: eark4_step(f, dt, param_dict, diss) )

    #Apply to update 
    f = jax.lax.fori_loop( 0, steps, update_f, f)
   
    return f




def vis(f):    
    figure, axis = plt.subplots(1,2)

    f = jnp.fft.irfft2(f)

    axis[0].imshow( f[0,:,:].transpose(), vmin=-10, vmax=10, cmap='bwr', origin='lower')
    axis[0].set_title("vorticity")
    
    axis[1].imshow( f[1,:,:].transpose(), vmin=-10, vmax=10, cmap='bwr', origin='lower' )
    axis[1].set_title("current")
    return figure, axis
















if (__name__ == "__main__"):
    print( f"Running {__file__} as main. Starting testing..." )

    #Load benchmarking and visualization modules
    import time

    # If this file is run as a standalone, we should do benchmarking.
    # Simulation parameters
    n = 128
    precision = jnp.float64
    # If you want double precision, change JAX defaults
    if (precision == jnp.float64):
        jax.config.update("jax_enable_x64", True)

    # Construct a dictionary for grid information
    param_dict = construct_domain(n, precision)

    # Get grids
    x = param_dict['x']
    y = param_dict['y']

    nu  = 1/100
    eta = 1/100
    b0  = [0.0, 0.1]  # Mean magnetic field
    forcing = -4*jnp.cos(4*y)

    # Append the extra system information to param_dict
    param_dict.update( {'nu': nu, 'eta': eta, 'b0': b0, 'forcing': forcing} )

    # Initial data
    f = jnp.zeros([2, n, n], dtype=precision)
    f = f.at[0, :, :].set(jnp.cos(x-0.1)*jnp.sin(x+y-1.2) + jnp.sin(3*x-1)*jnp.cos(y-1))
    f = f.at[1, :, :].set(jnp.cos(x+2.1)*jnp.sin(y+3.5))
    
    #fft the data before we evolve
    f = jnp.fft.rfft2(f)

    # number of times we evaluate the state velocity for benchmarking
    trials = 128

    v = lambda f: state_vel(f, param_dict, include_dissipation=False)

    
    start = time.time()
    for _ in range(trials):
        dfdt = v(f)
    stop = time.time()
    no_jit_time = stop-start
    print(f"state_vel {trials} times: {no_jit_time} seconds (no jit)")

    jit_v = jax.jit(v)
    _ = jit_v(f) #compile on first function call

    start = time.time()
    for _ in range(trials):
        dfdt = jit_v(f)
    stop = time.time()
    with_jit_time = stop-start
    print(f"state_vel {trials} times: {with_jit_time} seconds (with jit)")
    print(f"jit provided a x{no_jit_time/with_jit_time} speedup\n")
    



    print(f"Benchmarking DNS...")
    
    dt = 0.01
    steps = 1024*2

    start = time.time()
    _ = eark4(f, dt, steps, param_dict)
    stop = time.time()
    print(f"{steps} timesteps at dt={dt} took {stop-start} seconds (no jit).")


    
    #Try compiling it
    jit_eark4 = jax.jit(eark4)
    _ = jit_eark4(f, dt, steps, param_dict)

    start = time.time()
    f_final = jit_eark4(f, dt, steps, param_dict)
    stop = time.time()
    print(f"{steps} timesteps at dt={dt} took {stop-start} seconds (with jit).")


    #Make a figure of the vorticity and current after time evolution    
    figure, axis = vis(f_final)
    figure.savefig("figures/test.png", dpi=1000)
