def loss_RPO_memory_efficient( input_dict, param_dict, segments ):
    '''
    PURPOSE:
    loss_RPO has a critical flaw: poor memory scaling. This version is meant to give equivalent results, 
    but with a more manual approach to gradient calculation to prevent memory costs from exploding.
    '''

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
            return jnp.mean( jnp.square( f - f0 ) + jnp.square( v - v0 ) )



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

    '''
    checkpoints = [embed(input_dict)] * (segments + 1)
    # Step 2: integrate each segment
    for i in range(segments):
        checkpoints[i+1] = integrate(checkpoints[i])
    '''

    #Create a new dictionary with a batch axis
    checkpoints = extend_dictionary( embed(input_dict), segments+1 )  
    def update_checkpoints(i, checkpoints):
        next_state = integrate(restrict_dictionary(checkpoints, i))
        return update_dictionary_instance(checkpoints, i+1, next_state)
    
    checkpoints = jax.lax.fori_loop( lower=0, upper=segments, body_fun=update_checkpoints, init_val=checkpoints )
    
    # Step 3: compute the loss from final checkpoint
    #loss = criterion( checkpoints[-1], param_dict )
    loss = criterion( restrict_dictionary(checkpoints, -1), param_dict )
  
    return loss, input_dict

    ###########################################
    # Backwards evaluation of the gradient
    ###########################################

    # Anti step 3
    grad_criterion = jax.grad( criterion )
    grad = grad_criterion( checkpoints[-1], param_dict )

    # Anti step 2: transpose Jacobian products.
    # Abuse that we can write this as the gradient of a dot product.
    dict_dot   = lambda dict1, dict2: sum(jnp.vdot(dict1[k], dict2[k]) for k in dict1)
    math_trick = lambda dict1, dict2: dict_dot( integrate(dict1), dict2 )
    jacobian_transpose = jax.jit(jax.grad( math_trick ))
    for i in reversed(range(segments)):
        grad = jacobian_transpose(checkpoints[i], grad)
    
    #Anti step 1
    dict_dot   = lambda dict1, dict2: sum(jnp.vdot(dict1[k], dict2[k]) for k in dict1)
    math_trick = lambda dict1, dict2: dict_dot( embed(dict1), dict2 )
    jacobian_transpose = jax.grad( math_trick )  
    grad = jacobian_transpose(checkpoints[0], grad)

    return loss, grad