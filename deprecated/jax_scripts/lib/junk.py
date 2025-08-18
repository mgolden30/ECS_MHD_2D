def pack_RPO( f, T, sx):
    '''
    PURPOSE:
    Stack all info of a Relative Periodic Orbit(RPO) into a dictionary
    '''

    input_dict = {'fields': f, 'T': T, 'sx': sx}
    return input_dict

def unpack_RPO( z ):
    '''
    PURPOSE:
    Stack all info of a Relative Periodic Orbit(RPO) into a vector
    '''

    n = 256

    f  = jnp.reshape( z[0:2*n*n], [2,n,n] )
    #f = jax.lax.dynamic_slice(z, (0,), (2*256*256,) )
    f = jnp.reshape( f, [2,n,n] )
    
    T  = z[-2]
    sx = z[-1]
    
    return f, T, sx


def pack_RPO_multi( f, T, sx, n: int, segments: int):
    '''
    PURPOSE:
    Stack all info of a Relative Periodic Orbit(RPO) into a vector
    '''

    z = jnp.zeros([n*n*2*segments+2])
    z = z.at[:-2].set(jnp.reshape(f,-1))
    z = z.at[-2].set(T)
    z = z.at[-1].set(sx)
    
    return z

def unpack_RPO_multi( z ):
    '''
    PURPOSE:
    Stack all info of a Relative Periodic Orbit(RPO) into a vector
    '''

    #hardcoded resolution
    n = 256
    segments = 32

    f  = jnp.reshape( z[0:2*segments*n*n], [segments,2,n,n] )
    #f = jax.lax.dynamic_slice(z, (0,), (2*256*256,) )
    #f = jnp.reshape( f, [2,n,n] )
    
    T  = z[-2]
    sx = z[-1]
    
    return f, T, sx

def pack_RPO_multi_v2( f, Ts, sxs, n: int, segments: int):
    '''
    PURPOSE:
    Stack all info of a Relative Periodic Orbit(RPO) into a vector
    '''

    big = 2*n*n*segments
    z = jnp.zeros([big + 2*segments])
    z = z.at[0:big].set(jnp.reshape(f,-1))
    z = z.at[big:(big+segments)].set(Ts)
    z = z.at[(big+segments):(big+2*segments)].set(sxs)
    
    return z

def unpack_RPO_multi_v2( z ):
    '''
    PURPOSE:
    Stack all info of a Relative Periodic Orbit(RPO) into a vector
    '''

    #hardcoded resolution
    n = 256
    segments = 32

    big = 2*segments*n*n

    f  = jnp.reshape( z[0:big], [segments,2,n,n] )
    #f = jax.lax.dynamic_slice(z, (0,), (2*256*256,) )
    #f = jnp.reshape( f, [2,n,n] )
    
    Ts = z[big:(big+segments)]
    sxs = z[(big+segments):(big + 2*segments)]
    
    return f, Ts, sxs

def rk4_step(f, dt, param_dict):
    '''
    A step of 4th order Runge-Kutta.
    '''
    
    vel = lambda f: dt * state_vel(f, param_dict)
    
    k1 = vel(f)
    k2 = vel(f + k1/2)
    k3 = vel(f + k2/2)
    k4 = vel(f + k3)
    
    f = f + (k1 + 2*k2 + 2*k3 + k4)/6
    
    return f


def loss_traveling_wave( input_dict, param_dict ):
    '''
    PURPOSE:
    Compute a loss function for 
    '''

    f = input_dict['fields']
    f = jnp.fft.rfft2(f)

    dfdt = state_vel(f, param_dict )

    dfdx = f * 1j * param_dict['kx']
    
    wave_speed = input_dict['wave_speed']

    loss = dfdt - wave_speed * dfdx
    loss = jnp.fft.irfft2(loss)

    loss = jnp.mean( jnp.square(loss) )
    return loss

def loss_RPO( input_dict, param_dict ):
    '''
    Return a scalar loss for RPO
    '''

    # Unpack tensors we need 
    f = input_dict['fields']
    T = input_dict['T']
    sx= input_dict['sx']
    steps= param_dict['steps']

    f  = jnp.fft.rfft2(f)
    f0 = jnp.copy(f)

    dt = T/steps

    rk4_step_jit = jax.jit(rk4_step)
    update_f = lambda i, f: rk4_step_jit(f, dt, param_dict)
    
    f = jax.lax.fori_loop( 0, steps, update_f, f)

    # translate f0 along x
    f0 = f0 * jnp.exp(1j * param_dict['kx'] * sx)

    #compute the mismatch
    diff = f - f0
    diff = jnp.fft.irfft2(diff)

    loss = jnp.mean( jnp.square(diff) )
    return loss



def loss_RPO_multi( z, steps, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, n, seg ):
    '''
    Return a scalar loss for RPO

    Pick a particular segment seg to evaluate so we don't blow up our memory costs
    '''

    # Unpack the state
    f_all, T, sx = unpack_RPO_multi(z)

    segments = 32
    ip = (seg+1) % segments  # i + 1, index of next segment

    f = f_all[seg,  :, :, :]  # integrate this one forward in time
    f0 = f_all[ip, :, :, :]  # compare to this one

    f = jnp.fft.rfft2(f)
    f0 = jnp.fft.rfft2(f0)

    steps2 = round(256/32)
    h = T/steps2

    rk4_step_jit = jax.jit(rk4_step)
    update_f = lambda i, f: rk4_step_jit(f, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, h)
    #for _ in range(steps2):
    #    f = rk4_step_jit(f, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, h)
    f = jax.lax.fori_loop( 0, steps2, update_f, f)

    # translate f0
    f0 = f0 * jnp.exp(1j * kx * sx)

    diff = f - f0
    diff = jnp.fft.irfft2(diff)

    loss = jnp.mean( jnp.square(diff) )
    return loss

def loss_RPO_multi_v2( z, steps, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, n, seg ):
    '''
    Return a scalar loss for RPO

    Pick a particular segment seg to evaluate so we don't blow up our memory costs
    
    v2 - I let each segement have its own shooting time T and shift sx. This should help decouple the system and allow local relaxation.
    '''



    # Unpack the state
    f_all, Ts, sxs = unpack_RPO_multi_v2(z)

    segments = 32
    ip = (seg+1) % segments  # i + 1, index of next segment

    f = f_all[seg,  :, :, :]  # integrate this one forward in time
    f0 = f_all[ip, :, :, :]  # compare to this one

    f = jnp.fft.rfft2(f)
    f0 = jnp.fft.rfft2(f0)

    steps2 = round(256/32)
    h = Ts[seg]/steps2

    rk4_step_jit = jax.jit(rk4_step)
    update_f = lambda i, f: rk4_step_jit(f, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, h)
    #for _ in range(steps2):
    #    f = rk4_step_jit(f, kx, ky, to_u, to_v, mask, nu, eta, forcing, b0, h)
    f = jax.lax.fori_loop( 0, steps2, update_f, f)

    # translate f0
    f0 = f0 * jnp.exp(1j * kx * sxs[seg])

    diff = f - f0
    diff = jnp.fft.irfft2(diff)

    loss = jnp.mean( jnp.abs(diff) )
    return loss




def objective_RPO( input_dict, param_dict ):
    '''
    PURPOSE:
    Compute a vector f which will be zero when an RPO is passed.

    INPUT:
    input_dict - {'fields': fields, 'T': T, 'sx': sx}

    OUTPUT:
    f - a large column vector describing how much we fail to be a RPO
    '''

    #Unpack our dicts. Assume this is for a multishooting code
    fields = input_dict['fields']
    T  = input_dict['T'] #period per segment
    sx = input_dict['sx'] #shift per segment

    steps = param_dict['steps'] #timesteps per segment
    dt = T/steps

    segments = fields.shape[0]
    

    #Take the Fourier transform of our fields
    fields = jnp.fft.rfft2(fields)

    #Define a spatial shift operator
    shift = jnp.exp( 1j*param_dict['kx']*sx )

    #Define an update function for rk4 with a dummy variable for jax.lax.fori_loop
    update = lambda _, fields: rk4_step(fields, dt, param_dict)
    jit_update = jax.jit( update) 

    mismatches = ()

    #Loop over segments and look at local mismatch from forward time integration
    for i in range(segments):
        start = fields[i,:,:,:]
        ip = (i+1) % segments
        stop  = fields[i,:,:,:]

        #Forward time integration
        start = jax.lax.fori_loop(0, steps, jit_update, start)
        mismatch = start - shift*stop

        mismatches = mismatches + (mismatch,)
    

    '''
    mini_loss = lambda i, mismatches: mismatches + ( jax.lax.fori_loop(0, steps, jit_update, fields[i,:,:,:]) - shift*fields[(i+1)%segments,:,:,:], )

    #Replace the above with a call to fori_loop
    mismatches = jax.lax.fori_loop(0, segments, mini_loss, mismatches)
    '''
    
    #Concat the mismatches of each segment into a single tensor
    objective = jnp.stack(mismatches, axis=0) 

    # To make the system square, let's enforce two phase conditions for T and sx
    # Enforce these phase conditions on the first segment initial condition
    start = fields[0,:,:,:]

    #Phase in x: enforce that a certain Fourier mode of the flow is purely real.
    phase_x = jnp.imag( start[0,1,0] ) # the <1,0> mode of vorticity

    #For time, enforce d/dt(enst) = 0
    vel = state_vel(start, param_dict)

    #BAck to real space for dot product. Eventually remove this.
    start = jnp.fft.irfft2(start)
    vel   = jnp.fft.irfft2(vel)
    
    phase_t = jnp.mean( vel[0,:,:] * start )

    #Stack these all up into a single tensor. First transform the mismatches in Fourier space back to real space
    objective = jnp.fft.irfft2( objective )
    objective = jnp.reshape(objective, [-1])
    phase_x = 0*jnp.reshape(phase_x, [1]) #Make arrays instead of value
    phase_t = 0*jnp.reshape(phase_t, [1])
    
    #stack these 1d vectors into a big 1d vector
    f = jnp.concatenate( (objective, phase_x, phase_t) )
    return f




