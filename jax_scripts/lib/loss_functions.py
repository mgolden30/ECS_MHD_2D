import jax
import jax.numpy as jnp

import lib.mhd_jax as mhd_jax


def loss_RPO( input_dict, param_dict ):
    '''
    PURPOSE:
    Define a scalar loss for Relative Periodic Orbits (RPOs)
    '''

    # Unpack tensors we need 
    f  = input_dict['fields']
    T  = input_dict['T']
    sx = input_dict['sx']
    steps= param_dict['steps']

    f  = jnp.fft.rfft2(f)
    f0 = jnp.copy(f)

    dt = T/steps
    f = mhd_jax.eark4(f, dt, steps, param_dict )

    #Shift the resulting fields
    f = jnp.exp( -1j * param_dict['kx'] * sx ) * f

    #compute the mismatch
    diff   = f - f0
    diff_v = mhd_jax.state_vel(f, param_dict, include_dissipation=True) - mhd_jax.state_vel(f0, param_dict, include_dissipation=True)

    #Transform back to real space
    diff   = jnp.fft.irfft2(diff)
    diff_v = jnp.fft.irfft2(diff_v)

    #MSE error
    #loss = jnp.mean( jnp.square(diff)) + jnp.mean( jnp.square(diff_v) ) 
    loss = jnp.mean( jnp.abs(diff)) + jnp.mean( jnp.abs(diff_v) ) 

    return loss



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
    f = mhd_jax.eark4(f, dt, steps, param_dict )

    #Shift the resulting fields
    f = jnp.exp( -1j * param_dict['kx'] * sx ) * f

    #compute the mismatch
    diff   = f - f0

    #Transform back to real space
    diff   = jnp.fft.irfft2(diff)

    #Return a dictionary
    output_dict = {'fields': diff, 'T': 0.0, 'sx': 0.0}

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
            return jnp.mean( jnp.square( f - f0 ))# + jnp.square( v - v0 ) )
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