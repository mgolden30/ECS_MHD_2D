import jax.flatten_util
import jax.numpy as jnp
import jax 
import time

from scipy.io import savemat, loadmat
import lib.mhd_jax as mhd_jax


def dissipation_preconditioner( input_dict, param_dict, ravel_fn ):
    '''
    The objective function for a RPO is 

    F = exp(-i kx * sx) * u(T) - u(0)

    where u represents the whole MHD state. For high frequencies, the nonlinear terms of evolution are likely dominated by dissipation + advection from mean magnetic field.
    To help GMRES get the right answer for these high frequencies, we can analytically invert the Jacobian assuming such an evolution. In fourier space, the operator becomes

    J exp(ik.x) = [exp(-i kx*sx - nu k^2 T) - 1] exp(ik.x)

    where k.x = kx*x + ky*y.
    Let's just divide by this complex number and see if it is a nice preconditioner.
    '''

    kx = param_dict['kx']
    ky = param_dict['ky']

    k_sq = kx*kx + ky*ky

    nu  = param_dict['nu']
    eta = param_dict['eta']

    sx = input_dict['sx']
    T = input_dict['T']

    alpha_u = jnp.exp( -1j*kx*sx -  nu*T*k_sq ) - 1
    alpha_B = jnp.exp( -1j*kx*sx - eta*T*k_sq ) - 1

    #This is ugly. Rewrite eventually.
    alpha_u = jnp.reshape( alpha_u, [1,alpha_u.shape[0], alpha_u.shape[1]] )
    alpha_B = jnp.reshape( alpha_B, [1,alpha_B.shape[0], alpha_B.shape[1]] )

    #Stack the alphas
    alpha = jnp.concat( (alpha_u, alpha_B), axis=0)

    safe_alpha = jnp.where(param_dict['mask'], alpha, 1.0)  # avoid zero division
    div_alpha = jnp.where(param_dict['mask'], 1.0 / safe_alpha, 0.0)

    def divide_by_alpha( z ):
        #Assume z is a numerical vector. Turn it into an interpretable dict
        input_dict = ravel_fn(z)

        #FFT the fields
        f = input_dict['fields']
        f = jnp.fft.rfft2(f)
        f = f * div_alpha
        f = jnp.fft.irfft2(f)
        input_dict['fields'] = f

        #Turn back into a vector
        Mz = jax.flatten_util.ravel_pytree(input_dict)[0]
        return Mz

    return lambda z: divide_by_alpha(z)




def linear_dynamics_preconditioner( input_dict, param_dict, ravel_fn ):
    '''
    This is meant to be a very powerful preconditioner for RPO search that exactly inverts the Jacobian of the RPO objective function WITH PHASE CONSTRAINTS
    assuming the tangent vectors only evolve with dissipation and Alfven velocity from mean magnetic field.
    '''

    #At this moment, I only plan on supporting equal dissipations because it makes the math of Alfven waves easier
    assert( param_dict['nu'] == param_dict['eta'] )

    # Step 1: We think of the matrix we are inverting as having the structure
    #
    #
    #  J = [  A, u1, u2;
    #         v1, 0,  0;
    #         v2, 0,  0 ];
    #  
    #  In MATLAB notation. The first step is to construct A, u1, u2, v1, and v2 so we can go about inverting them.

    def construct_inv_A( ):
        # input_dict['fields'] is a [2,n,n] tensor. 
        # I will use jnp.fft.rfft2() to turn this into a [2,n,n//2+1] complex tensor of Fourier coefficients.
        # Then, the action of A is diagonal in Fourier space, but allows mixing of fluid / magnetic perturbations from Alfven waves.
        
        T  = input_dict['T']
        sx = input_dict['sx']

        kx = param_dict['kx']
        ky = param_dict['ky']
        
        k_sq = kx*kx + ky*ky
        mean_B = param_dict['b0']

        #Currently assuming nu == eta
        nu = param_dict['nu']

        #Construct the exponential that captures translation and dissipation
        e = jnp.exp( -1j * param_dict['kx'] * sx ) * jnp.exp( -nu * k_sq * T )

        #Construct sin and cos of the Alfven transport quantity
        alfven = (mean_B[0] * kx + mean_B[1] * ky) * T
        c = jnp.cos(alfven)
        s = jnp.sin(alfven)

        # Rather than deal with matrix multiplication, just
        # think of inv(A) as a highly symmetric 2x2 matrix 
        # 
        # inv(A) = [ Ai_11 , Ai_12;
        #            Ai_12 , Ai_11 ]
        Ai_11 = (e*c-1) / (e*e - 2*e*c + 1)
        Ai_12 = -1j* e * s / (e*e - 2*e*c + 1)

        #Mask values that are not of interest
        mask = param_dict['mask']
        Ai_11 = jnp.where( mask, Ai_11, 0 )
        Ai_12 = jnp.where( mask, Ai_12, 0 )

        #Define a function that applies this inverse matrix to a [2,n,n] tensor and returns a [2,n,n] tensor
        def apply_inv_A( f ):
            f = jnp.fft.rfft2(f)

            #apply inv(A) to hydro/magnetic components
            f1 = Ai_11 * f[0,:,:] + Ai_12 * f[1,:,:]
            f2 = Ai_12 * f[0,:,:] + Ai_11 * f[1,:,:]

            #Stack to get [2,n,n//2+1] tensor
            Ai_f = jnp.stack( [f1, f2], axis=0 )

            #invert fft
            Ai_f = jnp.fft.irfft2(Ai_f)
            return Ai_f
        
        return lambda f: apply_inv_A(f)
    
    def generate_u_and_v_vectors():
        #In my description above, we need four vectors u1, u2, v1, v2 to form the full system of equations
        
        #State
        f = input_dict['fields']
    
        #All dynamics functions in mhd_jax expect f to be passed in Fourier space.
        f = jnp.fft.rfft2(f)

        #v1 is just the full state velocity at t=0
        v1 = mhd_jax.state_vel( f, param_dict, include_dissipation=True )
        #v2 is the x derivative of the state at t=0
        v2 = 1j * param_dict['kx'] * f

        #The vectors u1 and u2 require integrating forward in time
        T  = input_dict['T']
        sx = input_dict['sx']
        steps= param_dict['steps']
        dt = T/steps
        f = mhd_jax.eark4(f, dt, steps, param_dict )

        #Shift the resulting fields
        f = jnp.exp( -1j * param_dict['kx'] * sx ) * f

        #u1 is the velocity of the shifted final state
        u1 = mhd_jax.state_vel( f, param_dict, include_dissipation=True )
        #u2 is the derivative with respect to sx, which is the negative x derivative
        u2 = - 1j* param_dict['kx'] * f

        #Put these back in real space before returning
        u1 = jnp.fft.irfft2(u1)
        u2 = jnp.fft.irfft2(u2)
        v1 = jnp.fft.irfft2(v1)
        v2 = jnp.fft.irfft2(v2)
        
        return u1, u2, v1, v2

    #Do the "intense" numerics once up front.
    u1, u2, v1, v2 = generate_u_and_v_vectors()
    apply_inv_A = construct_inv_A()

    #Define a function to invert our linear system so we can return an anonymous function for it.
    def invert_linear_system( z ):
        #z is a 1D vector. Convert it to an interpretable dictionary.
        input_dict = ravel_fn(z)

        #unpack the state into tensors
        f  = input_dict['fields']
        T  = input_dict['T']
        sx = input_dict['sx']

        #Build the 2x2 matrix
        M = jnp.zeros([2,2])
        b = jnp.zeros([2,])
        
        Ai_f  = apply_inv_A(f)
        Ai_u1 = apply_inv_A(u1)
        Ai_u2 = apply_inv_A(u2)

        M = M.at[0,0].set( jnp.sum( v1 * Ai_u1 ) )
        M = M.at[0,1].set( jnp.sum( v1 * Ai_u2 ) )
        M = M.at[1,0].set( jnp.sum( v2 * Ai_u1 ) )
        M = M.at[1,1].set( jnp.sum( v2 * Ai_u2 ) )

        b = b.at[0].set( jnp.sum(v1 * Ai_f) )
        b = b.at[1].set( jnp.sum(v2 * Ai_f) )

        #Solve the 2x2 system with lsqr
        x, _, _, _ = jnp.linalg.lstsq(M,b)

        #Use this output to solve the general problem
        f_out = apply_inv_A( f - x[0]*u1 - x[1]*u2 )

        output_dict = {"fields": f_out, "T": x[0], "sx": x[1] }

        return jax.flatten_util.ravel_pytree(output_dict)[0]

    return lambda z: invert_linear_system(z)





def diagonal_preconditioner_spatial(input_dict, param_dict, jac, k=8, batch=4):
    print("\nEstimating diagonal of Jacobian...")

    #Size of our random vectors 
    n = input_dict['fields'].size
    diag = jnp.zeros([n,])

    def random_vector(key, size):
        # Generates a vector with random ±1 values
        return (2 * jax.random.bernoulli(key, 0.5, shape=size).astype(jnp.float64) - 1.0)
    
    #Define the linear operator of interest
    _, unravel_fn_right = jax.flatten_util.ravel_pytree(input_dict)
    lin_op = lambda v:  jax.flatten_util.ravel_pytree(  jac( input_dict, unravel_fn_right(v))  )[0]

    #Why wait for each matrix-vector product individually?
    lin_op_batched = jax.jit(jax.vmap(lin_op))

    #Use k random vectors to estimate diagonal
    key = jax.random.PRNGKey(seed=0)

    for i in range(k):
        #Split the key
        key, subkey = jax.random.split(key)
        z = random_vector(subkey, (batch, n) )

        #The matrix is n-by(n+2), so pad with a zero on both ends.
        z_pad = jnp.pad( z, ((0,0),(1,1)), mode='constant' )

        print(z_pad.shape)

        #Apply the Jacobian and take the product
        start = time.time()
        diag += jnp.sum( z * lin_op_batched(z_pad), axis=0 )
        stop = time.time()
        
        print(f"{i}: walltime of {stop-start} seconds.")

    diag = diag/k/batch
    
    #Save for investigation
    savemat("spatial_diag.mat",{"diag": jnp.reshape(diag, input_dict['fields'].shape)})    

    def M(v, mode):
        '''
        INPUT:
        v - a vector
        mode - "trans" or "no_trans"

        OUTPUT:
        Mv - the matrix-vector product. Since M is self adjoint here, mode is ignored.
        '''

        #Multiply. Both fields have been pre-dealiased
        Mv =  v/ diag

        #Dealias again after pointwise multiplication
        #Mv = jnp.fft.rfft2(Mv)
        #Mv = param_dict['mask'] * Mv
        #Mv = jnp.fft.irfft2(Mv)
        return Mv
    return M




def diagonal_preconditioner_fourier( input_dict, jac, k=8, batch=4 ):
    '''
    Estimate the diagonal of a linear operator in the Fourier representation.
    '''

    print("\nEstimating diagonal of Jacobian...")

    #Size of our random vectors 
    n = input_dict['fields'].size
    diag = 0

    def random_vector(key, shape):
        # Generate a real-valued i.i.d. Gaussian field
        real_field = jax.random.normal(key, shape)
        # Return its Fourier transform (Hermitian-symmetric)
        fourier = jnp.fft.rfft2( real_field )
        # Just the phases, unit amplitude
        fourier = fourier / jnp.abs(fourier)
        return fourier
    
    #Define the linear operator of interest
    _, unravel_fn_right = jax.flatten_util.ravel_pytree(input_dict)
    lin_op = lambda v:  jax.flatten_util.ravel_pytree(  jac( input_dict, unravel_fn_right(v))  )[0]

    #Why wait for each matrix-vector product individually?
    lin_op_batched = jax.jit(jax.vmap(lin_op))

    #Use k random vectors to estimate diagonal
    key = jax.random.PRNGKey(seed=0)
    for i in range(k):
        #Split the key
        key, subkey = jax.random.split(key)

        #Generate random vectors in Fourier space
        z = random_vector(subkey, (batch,) + input_dict['fields'].shape )

        #Transform them to real space
        z_real = jnp.fft.irfft2(z)
        z_real = jnp.reshape( z_real, [batch, n] )
        
        #The matrix is acutally n-by(n+2), so pad with zeros on both ends.
        z_pad = jnp.pad( z_real, ((0,0),(1,1)), mode='constant' )

        #Apply the Jacobian and take the product
        start = time.time()
        Jz = lin_op_batched(z_pad)
        Jz = jnp.reshape( Jz, (batch,) + input_dict['fields'].shape )
        Jz = jnp.fft.rfft2(Jz)
        diag += jnp.sum( jnp.conj(z) * Jz, axis=[0] )
        stop = time.time()
        
        print(f"{i}: walltime of {stop-start} seconds.")

    #Divide by the number of vectors we sampled
    diag = diag/k/batch

    #Transform to real space
    diag = jnp.fft.irfft2(diag)

    savemat("fourier_diag.mat",{"diag": jnp.reshape(diag, input_dict['fields'].shape) })
    
    #Prevent division by zero
    #diag = 1 + jnp.abs(diag)
        
    def M(v, mode):
        #Diagonal matrices are self-adjoint.
        #mode (which will be "trans" or "no_trans") is purposefully ignored
        return v/diag
    return M




def floquet_preconditioner( filename, epsilon ):
    '''
    Precondition with the leading Floquet spectrum. This is only sensible if you are close 
    to a solution in the first place.
    '''

    data = loadmat(filename)
    
    #Read in the leading Schur vectors
    Q = data['tang']
    Q = jnp.reshape(Q, [Q.shape[0], -1] )

    print(Q.shape)

    #Read in the approximate Schur form of the matrix.
    #This is only close to upper triangular. Be prepared for it to be dense.
    R = data['R']

    #Subtract the identity from R
    R2 = R - jnp.identity(R.shape[0])
    
    #Compute the psuedo-inverse with some tolerance
    #R2_inv = jnp.linalg.pinv( R2, rtol=1e-6 )

    U, S, Vh = jnp.linalg.svd(R2, full_matrices=False)
    S_inv = 1.0 / (S + epsilon)
    R2_inv = Vh.T @ (S_inv[:, None] * U.T)


    def M(v, mode):
        #project the vector into the approximate Schur vectors
        u = Q @ v
            
        #Determine the leftover, perpindicular part of v to our Schur vectors
        v_perp = v - Q.transpose() @ u

        #What we do with u depends on the evaluation mode
        match mode:
            case "no_trans":
                #Multiply by our pseudoinverse of R-I
                u = R2_inv @ u
            case "trans":
                #Multiply by transpose of our pseudoinverse of R-I
                 u = R2_inv.transpose() @ u
            case _:
                print(f"Oh no. You passed mode = {mode} to the preconditioner. This is not a valid option...")
                return None #just crash out everything
        #Recombine with v_perp and return
        Mv = v_perp + Q.transpose() @ u 
        return Mv
    return M

