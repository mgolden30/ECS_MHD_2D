import jax
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax
import lib.timestepping as timestepping
import lib.symmetry as symm


def mismatch_RPO( input_dict, param_dict, mode ):
    """
    PURPOSE:
    Compute the mismatch between initial and final conditions for an RPO.

    The purpose of this function is to be a unified interface for various time integration schemes that I can use
    for both adjoint descent and Newton-Raphson iteration. I want this to be the only function in my code
    that does forward time integration for converging RPOs.

    Parameters
    ----------
    input_dict : dictionary
        A dictionary of tensors I want to optimize. Should contain "fields", "T", and "sx".
    param_dict : dictionary
        A dictionary of all other information needed by the integrator.
    mode : string
        A flag that determines the integration called integration routine
    
    Returns
    -------
    f0 : array
        initial condition
    f : array
        final condition with symmetries applied. For an RPO, f = f0. 
        That is, f = apply(g, integrate(f0,T)) in pseudocode.
    info : Any
        Any information from integration you might want to return. 
        For adaptive timestepping, this might be useful.
    """

    # The tensor of input dict will be the optimized quantities
    f0 = input_dict['fields'] #initial condition (in real space) assumed of size [2,n,n]
    T  = input_dict['T']      #period
    sx = input_dict['sx']     #spatial shift (assuming continuous symmetry in x)
    
    #Do integration in Fourier space
    f0  = jnp.fft.rfft2(f0)

    #Do time integration depending on the user specified mode
    match mode:
        case "RK4":
            num_checkpoints = param_dict['num_checkpoints']
            ministeps = param_dict['ministeps']
            v_fn = lambda f: mhd_jax.state_vel(f,param_dict,include_dissipation=True)
            integrate = jax.checkpoint( lambda _, f : timestepping.rk4(f, T/num_checkpoints, ministeps, v_fn))
            f = jax.lax.fori_loop( 0, num_checkpoints, integrate, f0)
            info = None
        case "Lawson_RK4":
            num_checkpoints = param_dict['num_checkpoints']
            ministeps = param_dict['ministeps']
            v_fn = lambda f: mhd_jax.state_vel(f,param_dict,include_dissipation=False)
            L_diag = mhd_jax.dissipation(param_dict)
            integrate = jax.checkpoint( lambda _, f : timestepping.lawson_rk4(f, T/num_checkpoints, ministeps, v_fn, L_diag, mask=param_dict['mask']))
            f = jax.lax.fori_loop( 0, num_checkpoints, integrate, f0)
            info = None
        case "Lawson_RK6":
            v_fn = lambda f: mhd_jax.state_vel(f,param_dict,include_dissipation=False)
            L_diag = mhd_jax.dissipation(param_dict)
            f = timestepping.lawson_rk6(f0, t=T, steps=param_dict['steps'], v_fn=v_fn, L_diag=L_diag, mask=param_dict['mask'])
            info = None
        case "Lawson_RK43":
            #Provide a nonlinear velocity function
            #vel_fn = lambda f: mhd_jax.state_vel(f, param_dict, include_dissipation=False)    
            #Integrate with adaptive timestepping
            #f, info = timestepping.eark43(f0, vel_fn, dissipation, T, h=1e-2, atol=adaptive_dict["atol"], max_steps_per_checkpoint=adaptive_dict["max_steps_per_checkpoint"], checkpoints=adaptive_dict["checkpoints"] )
            f = f0
            info = None #Change this for adaptive timestepping
        case _:
            print(f"You selected mode = {mode}, which does not exist. Exiting...")
            exit()

    #Shift the resulting fields with symmetry operations.
    f = symm.shift_x(f, sx, param_dict)
    f = symm.shift_reflect(f, param_dict['shift_reflect_ny'], param_dict)
    if param_dict['rot']:
        f = symm.rot180(f)

    return f0, f, info


def loss_RPO( input_dict, param_dict, mode ):
    #Do forward time integration
    f0, f, info = mismatch_RPO( input_dict, param_dict, mode )

    #compute mismatch
    diff = f - f0

    #Transform to real space
    diff = jnp.fft.irfft2(diff)

    #Compute the MSE
    loss = jnp.mean(jnp.square(diff))
    return loss, info

def phase_conditions(f0, param_dict):
    #Compute the x and t derivatives for phase conditions
    vt = mhd_jax.state_vel(f0, param_dict, include_dissipation=True)
    vx = 1j*param_dict['kx']*f0

    #Move them to real space and stop the gradient
    vt = jax.lax.stop_gradient( jnp.fft.irfft2(vt) )
    vx = jax.lax.stop_gradient( jnp.fft.irfft2(vx) )
    
    #Compute dot products with our state in real space
    f0 = jnp.fft.irfft2(f0)
    pt = jnp.mean( vt * f0 )
    px = jnp.mean( vx * f0 )

    #Force them to be zero, but not differentiate to zero
    pt = pt - jax.lax.stop_gradient(pt)
    px = px - jax.lax.stop_gradient(px)
    return pt, px

def objective_RPO( input_dict, param_dict, mode ):
    '''
    PURPOSE:
    Define a vector objective for Relative Periodic Orbits (RPOs). We can use Newton methods to converge this.
    '''

    #Do forward time integration
    f0, f, info = mismatch_RPO( input_dict, param_dict, mode )
        
    #Phase conditions are a hack to implement orthogonality conditions with autodiff
    pt, px = phase_conditions(f0, param_dict)

    #Create a dictionary with identical names to input_dict
    out_dict = {"fields": jnp.fft.irfft2(f0 - f), "T": pt, "sx": px }

    #Discard info for now. TBD how I handle it.
    return out_dict




"""
def objective_RPO_adaptive( input_dict, param_dict, adaptive_dict ):
    '''
    PURPOSE:
    Define a scalar loss for Relative Periodic Orbits (RPOs)
    '''

    # Unpack tensors we need 
    f0 = input_dict['fields']
    T  = input_dict['T']
    sx = input_dict['sx']
    
    f0  = jnp.fft.rfft2(f0)

    #Construct a dissipation operator
    k_sq = param_dict['kx']**2 + param_dict['ky']**2
    coeffs = jnp.array([param_dict['nu'], param_dict['eta']])
    coeffs = jnp.reshape(coeffs, [2,1,1])
    dissipation = - k_sq * coeffs

    #Provide a nonlinear velocity function
    vel_fn = lambda f: mhd_jax.state_vel(f, param_dict, include_dissipation=False)
    
    #Integrate in time
    f, info = timestepping.eark43(f0, vel_fn, dissipation, T, h=1e-2, atol=adaptive_dict["atol"], max_steps_per_checkpoint=adaptive_dict["max_steps_per_checkpoint"], checkpoints=adaptive_dict["checkpoints"] )

    #Shift the resulting fields
    f = symm.shift_x(f, sx, param_dict)
    f = symm.shift_reflect(f, param_dict['shift_reflect_ny'], param_dict)
    if param_dict['rot']:
        f = symm.rot180(f)
    
    #compute the mismatch
    diff = jnp.fft.irfft2(f - f0)
    
    #Compute the x and t derivatives for phase conditions
    vt = mhd_jax.state_vel(f0, param_dict, include_dissipation=True)
    vx = 1j*param_dict['kx']*f0

    #Move them to real space and stop the gradient
    vt = jax.lax.stop_gradient( jnp.fft.irfft2(vt) )
    vx = jax.lax.stop_gradient( jnp.fft.irfft2(vx) )
    
    #Compute dot products with our state
    f0 = jnp.fft.irfft2(f0)
    pt = jnp.sum( vt * f0 )
    px = jnp.sum( vx * f0 )

    #Force them to be zero, but not differentiate to zero
    pt = pt - jax.lax.stop_gradient(pt)
    px = px - jax.lax.stop_gradient(px)

    #Try rescaling these constraints to see the impact
    pt = pt / f0.size
    px = px / f0.size

    #Create a dictionary with identical names to input_dict
    out_dict = {"fields": diff*(1.0 + 1.0/(T*T)), "T": pt, "sx": px }

    #jax.debug.print(f"completed: {info['completed']}, accepted: {info['accepted']}, rejected: {info['rejected']}")
    return out_dict 
"""



""""
def objective_RPO_multishooting( input_dict, param_dict ):
    '''
    PURPOSE:
    Define a vector objective for Relative Periodic Orbits (RPOs). We can use Newton methods to converge this.
    This variant assumes the state is for multishooting
    '''

    #Map initial data to Fourier space
    f0  = jnp.fft.rfft2(input_dict['fields'])

    #Compute timestep and steps per segment, which I call ministeps
    #Here the period T and steps are defined for the WHOLE orbit
    dt = input_dict["T"]/param_dict["steps"]
    #ministeps = param_dict["ministeps"] #steps / segment
    miniministeps = param_dict["miniministeps"] #ministeps / checkpoint
    
    def make_eark4(steps):
        def eark4(f, dt, param_dict):
            '''
            Perform many steps of Exponential Ansatz Runge-Kutta 4 (EARK4)
            '''
            #Construct a diagonal dissipation operator
            diss_coeffs = jnp.array( [param_dict['nu'], param_dict['eta']] )
            diss_coeffs = jnp.reshape(diss_coeffs, [2,1,1])
            k_sq = jnp.square(param_dict['kx']) + jnp.square(param_dict['ky'] )
            diss = jnp.exp( - diss_coeffs  * k_sq * dt / 2 ) * param_dict['mask']
            
            #Need this lambda format to use fori_loop
            update_f = lambda _, f: mhd_jax.eark4_step(f, dt, param_dict, diss)
            f = jax.lax.fori_loop( 0, steps, update_f, f)            
            return f
        return eark4
    
    #Define an integration routine that checkpoints
    safe_eark4 = jax.checkpoint( make_eark4(miniministeps) )
    update_f = lambda _, f: safe_eark4(f, dt, param_dict)
    f = jax.lax.fori_loop( 0, param_dict["checkpoints"], update_f, f0)

    #Shift the resulting fields
    f = jnp.exp( -1j * param_dict['kx'] * input_dict['sx'] / param_dict['segments'] ) * f

    #compute the mismatch
    diff = jnp.fft.irfft2(f0 - jnp.roll(f, shift=1, axis=0))

    #Compute mismatch in velocity as well
    v  = mhd_jax.state_vel( f, param_dict, include_dissipation=True )
    v0 = mhd_jax.state_vel( f0, param_dict, include_dissipation=True ) 
    diff_vel = jnp.fft.irfft2( v0 - jnp.roll(v, shift=1, axis=0) )

    #Return a dictionary
    output_dict = {'fields': diff, 'vel': diff_vel}
    #output_dict = {'fields': diff }

    return output_dict


def add_orthogonal_contraints( objective_fn, param_dict, Q ):
    '''
    Return a new function that adds appends orthogonality constraints.
    '''
    def new_objective_fn( input_dict, param_dict ):
        #Evaluate the original function 
        output_dict = objective_fn(input_dict, param_dict)
        
        #Assume vectors is of size [m, n], where m is the number of vectors and 
        #n is the dimension of the input_dict
        u = jax.flatten_util.ravel_pytree( input_dict['fields'] )[0]

        #The dot product of our Q vectors with the field u
        c = Q @ u

        #This is some jax magic to set c to zero, but evaluate the Jacobian as Q @ du
        c = c - jax.lax.stop_gradient(c)

        #Append these phase constraints to your obejctive
        output_dict.update({"ortho": c})
        return output_dict
    return new_objective_fn





def objective_RPO( input_dict, param_dict ):
    '''
    PURPOSE:
    Define a vector objective for Relative Periodic Orbits (RPOs). We can use Newton methods to converge this.
    '''

    # Unpack tensors we need 
    f  = input_dict['fields']
    T  = input_dict['T']
    sx = input_dict['sx']
    steps= param_dict['steps']

    f  = jnp.fft.rfft2(f)
    f0 = jnp.copy(f)

    dt = T/steps

    #f = mhd_jax.eark4(f, dt, steps, param_dict )
    f = mhd_jax.lawson_rk6(f0, dt, steps, param_dict)
    
    #Shift the resulting fields
    f = symm.shift_x(f, sx, param_dict)
    f = symm.shift_reflect(f, param_dict['shift_reflect_ny'], param_dict)
    if param_dict['rot']:
        f = symm.rot180(f)
    
    #compute the mismatch
    diff = jnp.fft.irfft2(f - f0)
    
    #Compute the x and t derivatives for phase conditions
    vt = mhd_jax.state_vel(f0, param_dict, include_dissipation=True)
    vx = 1j*param_dict['kx']*f0

    #Move them to real space and stop the gradient
    vt = jax.lax.stop_gradient( jnp.fft.irfft2(vt) )
    vx = jax.lax.stop_gradient( jnp.fft.irfft2(vx) )
    
    #Compute dot products with our state
    f0 = jnp.fft.irfft2(f0)
    pt = jnp.sum( vt * f0 )
    px = jnp.sum( vx * f0 )

    #Force them to be zero, but not differentiate to zero
    pt = pt - jax.lax.stop_gradient(pt)
    px = px - jax.lax.stop_gradient(px)

    #Try rescaling these constraints to see the impact
    pt = pt / f0.size
    px = px / f0.size

    #Create a dictionary with identical names to input_dict
    out_dict = {"fields": diff*(1.0 + 1.0/(T*T)), "T": pt, "sx": px }

    #jax.debug.print(f"completed: {info['completed']}, accepted: {info['accepted']}, rejected: {info['rejected']}")
    return out_dict 

def objective_RPO_with_checkpoints( input_dict, param_dict ):
    '''
    PURPOSE:
    Define a vector objective for Relative Periodic Orbits (RPOs). 
    This variant has checkpointing so that jax.jvp doesn't run out of memory and I can use adjoint information for Newton.
    '''

    # Unpack tensors we need 
    f  = input_dict['fields']
    T  = input_dict['T']
    sx = input_dict['sx']

    ministeps = param_dict['ministeps']
    num_checkpoints = param_dict['num_checkpoints']
    dt = T/(ministeps * num_checkpoints)

    print(ministeps)
    print(type(ministeps))
    print(num_checkpoints)
    print(type(num_checkpoints))

    f  = jnp.fft.rfft2(f)
    f0 = jnp.copy(f)

    def make_eark4(steps):
        def eark4(f, dt, param_dict):
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
            update_f = lambda _, f: mhd_jax.eark4_step(f, dt, param_dict, diss)
            #Apply to update 
            f = jax.lax.fori_loop( 0, steps, update_f, f)
            
            #ChatGPT solution using scan
            #def step_fn(f, _):
            #    return mhd_jax.eark4_step(f, dt, param_dict, diss), None

            #f, _ = jax.lax.scan(step_fn, f, None, length=steps)
            return f
        return eark4
    
    #Define an integration routine that checkpoints
    safe_eark4 = jax.checkpoint( make_eark4(ministeps) )
    update_f = lambda _, f: safe_eark4(f, dt, param_dict)
    f = jax.lax.fori_loop( 0, num_checkpoints, update_f, f)

    #Shift the resulting fields
    f = jnp.exp( -1j * param_dict['kx'] * sx ) * f

    #compute the mismatch
    diff   = f - f0

    #Transform back to real space
    diff   = jnp.fft.irfft2(diff)

    #Return a dictionary
    output_dict = {'fields': diff}

    return output_dict


def add_phase_conditions( input_dict, tangent_dict, Jtangent_dict, param_dict ):
    '''
    PURPOSE:
    Add phase conditions to the action of the Jacobian
    '''
    
    f  = input_dict['fields']
    df = tangent_dict['fields']

    f    = jnp.fft.rfft2(f)
    dfdt = mhd_jax.state_vel( f, param_dict, include_dissipation=True )
    dfdx = 1j * param_dict['kx'] * f

    dfdt = jnp.fft.irfft2(dfdt)
    dfdx = jnp.fft.irfft2(dfdx)

    Jtangent_dict['T']  = jnp.sum( df * dfdt )
    Jtangent_dict['sx'] = jnp.sum( df * dfdx )
    return Jtangent_dict
"""



def loss_RPO_memory_efficient( input_dict, param_dict, segments ):
    '''
    PURPOSE:
    loss_RPO has a critical flaw: poor memory scaling. This version is meant to give equivalent results, 
    but with a more manual approach to gradient calculation to prevent memory costs from exploding.
    '''

    def embed( input_dict ):
        '''
        expand the input dictionary to contain 'fields0'.
        '''
        return {'fields': input_dict['fields'], 'T': input_dict['T'], 'sx': input_dict['sx'], 'fields0': jnp.copy(input_dict['fields']) }



    def criterion( input_dict, param_dict ):
            '''
            shift and return the whole 
            '''

            #Shift the resulting fields
            f = jnp.fft.rfft2( input_dict['fields'] )
            f = jnp.exp( -1j * param_dict['kx'] * input_dict['sx'] ) * f
            
            f0 = input_dict['fields0']
            f0 = jnp.fft.rfft2(f0)

            v  = mhd_jax.state_vel(f,  param_dict, include_dissipation=True )
            v0 = mhd_jax.state_vel(f0, param_dict, include_dissipation=True )

            f = jnp.fft.irfft2(f)
            f0= jnp.fft.irfft2(f0)
            v = jnp.fft.irfft2(v)
            v0= jnp.fft.irfft2(v0)

            #Compute the mean squared error
            return jnp.mean( jnp.square( f - f0 )) #+ jnp.square( v - v0 ) )
            #return jnp.mean( jnp.abs( f - f0 ) + jnp.abs( v - v0 ) )


    def integrate_segment( input_dict, param_dict, segments ):
            f  = input_dict['fields']
            T  = input_dict['T']

            steps = param_dict['steps'] // segments

            dt = T/steps/segments

            f = jnp.fft.rfft2(f)
            f = mhd_jax.eark4(f, dt, steps, param_dict )
            f = jnp.fft.irfft2(f)

            return {'fields': f, 'T': input_dict['T'], 'sx': input_dict['sx'], 'fields0': input_dict['fields0'] }

    def extend_dictionary( dictionary, batch ):
        """
        Repeat each tensor in the dictionary along a new leading batch dimension.

        Args:
        dictionary: A dict where each value is a jnp.ndarray.
        batch: Integer, the size of the new leading batch dimension.

        Returns:
        A new dict with each tensor repeated along the leading dimension.
        """
        return {k: jnp.broadcast_to(v, (batch,) + v.shape) for k, v in dictionary.items()}

    def restrict_dictionary(dictionary, n):
        """
        Restrict a dictionary of batched tensors to the nth instance along the batch axis.

        Args:
            dictionary: A dict where each value is a jnp.ndarray with a leading batch dimension.
            n: Integer index to extract from the batch dimension.

        Returns:
            A new dict with the same keys, where each value has shape value.shape[1:] (batch axis removed).
        """
        return {k: v[n] for k, v in dictionary.items()}


    def update_dictionary_instance(batched_dict, n, new_entry):
        """
        Update the nth instance in a batched dictionary with values from a new dictionary.

        Args:
            batched_dict: A dict where each value is a jnp.ndarray with leading batch dimension.
            n: Integer index to update along the batch axis.
            new_entry: A dict with the same keys, containing arrays of shape matching batched_dict[key].shape[1:].

        Returns:
            A new batched dict with the nth instance updated.
        """
        return {
            k: jax.lax.dynamic_update_index_in_dim(batched_dict[k], new_entry[k], n, axis=0)
            for k in batched_dict
        }
    

    ##################################
    # Forward evaluation of the loss
    ##################################

    # Step 1: append the initial field
    integrate = lambda input_dict: integrate_segment(input_dict, param_dict, segments)
    integrate = jax.jit(integrate)

    #Create a new dictionary with a batch axis
    checkpoints = extend_dictionary( embed(input_dict), segments+1 )  
    def update_checkpoints(i, checkpoints):
        next_state = integrate(restrict_dictionary(checkpoints, i))
        return update_dictionary_instance(checkpoints, i+1, next_state)
    
    #Do the forward time integration with a fori_loop in an attempt to get jit to work while respecting memory
    checkpoints = jax.lax.fori_loop( lower=0, upper=segments, body_fun=update_checkpoints, init_val=checkpoints )
    
    # Step 3: compute the loss and grad from final checkpoint
    crit = jax.value_and_grad(criterion) #we need both, might as well
    loss, grad = crit( restrict_dictionary(checkpoints, -1), param_dict )
  



    ###########################################
    # Backwards evaluation of the gradient
    ###########################################

    # Anti step 2: transpose Jacobian products.
    # Abuse that we can write this as the gradient of a dot product.
    dict_dot   = lambda dict1, dict2: sum(jnp.vdot(dict1[k], dict2[k]) for k in dict1)
    math_trick = lambda dict1, dict2: dict_dot( integrate(dict1), dict2 )
    jacobian_transpose = jax.jit(jax.grad( math_trick ))

    def update_grad(i, grad):
        i_reversed = segments-1-i
        return jacobian_transpose( restrict_dictionary(checkpoints, i_reversed), grad)
    
    #for i in reversed(range(segments)):
    #    grad = jacobian_transpose(checkpoints[i], grad)
    grad = jax.lax.fori_loop( 0, segments, update_grad, grad )

    #Anti step 1
    dict_dot   = lambda dict1, dict2: sum(jnp.vdot(dict1[k], dict2[k]) for k in dict1)
    math_trick = lambda dict1, dict2: dict_dot( embed(dict1), dict2 )
    jacobian_transpose = jax.grad( math_trick )  
    grad = jacobian_transpose(restrict_dictionary(checkpoints, 0), grad)

    #For some reason, grad has an entry for fields0
    del grad['fields0']

    return loss, grad


def traveling_wave_loss( input_dict, param_dict ):

    #Compute the state velocity
    f = jnp.fft.rfft2(input_dict['fields'])
    v = mhd_jax.state_vel( f, param_dict, include_dissipation=True)

    #Compute the x derivative
    fx = 1j * param_dict['kx'] * f
    c  = input_dict['wave_speed']

    #for a traveling wave, \partial_t = c \partial_x
    loss = v - c*fx

    loss = jnp.fft.irfft2(loss)
    loss = jnp.mean(jnp.square(loss))
    return loss



def traveling_wave_objective( input_dict, param_dict ):
    #Compute the state velocity
    f = jnp.fft.rfft2(input_dict['fields'])
    v = mhd_jax.state_vel( f, param_dict, include_dissipation=True)

    #Compute the x derivative
    fx = 1j * param_dict['kx'] * f
    c  = input_dict['wave_speed']

    #for a traveling wave, \partial_t = c \partial_x
    loss = v - c*fx

    loss = jnp.fft.irfft2(loss)
    return {"fields": loss}
