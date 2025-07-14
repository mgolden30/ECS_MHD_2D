
import jax
import jax.numpy as jnp

from scipy.io import savemat





def minimize_vector_quadratic(a, b, c, verbose=False):
    '''
    Find x that minimizes the 2-norm of the vector quadratic equation

    b + a*x + 1/2*\sum_n c_n*x_n^2,

    where b is a vector, a is a matrix, and c_n is a set of vectors. That is, the quadratic equation is "diagonal" in some sense.
    
    The assumption of this code  is that a is small, since x is a problematic direction.
    '''

    if verbose:
        print(f"Solving qudatic equation. Shape of b is {b.shape}.")

    #Define the function we minimize
    norm_b = jnp.linalg.norm(b)
    f = lambda x: (b + a @ x + 1/2 * c @ (x*x))/norm_b

    #Make initial guess assuming that "a" doesn't matter
    x, _, _, _ = jnp.linalg.lstsq( c, 2*b )
    x = x.at[x<0].set(0.0)
    x = jnp.sqrt(x)

    z = jnp.linalg.lstsq( c, -2*b, rcond=None)[0]
    x0 = jnp.sign(z) * jnp.sqrt(jnp.abs(z))  # guess x such that x*x ~ z

    #Fine tune with an optimizer
    loss_fn = lambda x: jnp.linalg.norm( f(x) )

    # ADAM hyperparameters
    learning_rate = 1e-2
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Optimization loop
    val_and_grad_fn = jax.value_and_grad(loss_fn)
    t = 1
    m = 0
    v = 0
    for step in range( 4*1024 ):
        t += 1
        val, g = val_and_grad_fn(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        if verbose and (step % 512 == 0):
            print(f"{step}: f = {val:.6}")

    return x






def solve_scalar_problem(a,b,c):
    '''
    Minimize the 2-norm of  ax + b + c x^2, where {a,b,c} are vectors and x is a scalar.
    We can do this by solving a cubic polynomial.
    '''

    aa = jnp.dot(a,a)
    ab = jnp.dot(a,b)
    ac = jnp.dot(a,c)
    bb = jnp.dot(b,b)
    bc = jnp.dot(b,c)
    cc = jnp.dot(c,c)
    
    residual = lambda x: bb + 2*ab*x + (aa + 2*bc)*x*x + 2*ac*x*x*x + cc*x*x*x*x
    
    #The gradient of the residual function is a cubic polynomial
    coeffs = jnp.array( [4*cc, 6*ac, 2*aa + 4*bc, 2*ab] )
    x = jnp.roots(coeffs)

    #Restrict to real roots
    is_real = jnp.abs( jnp.imag(x) ) < 1e-12
    x = jnp.real( x[is_real] )

    #Pick the smallest residual
    r = residual(x)
    i = jnp.argmin(r)

    return x[i], jnp.sqrt(r[i])



def kaczmarz_iteration( a, b, c, maxit ):
    '''
    Minimize the 2-norm of ax + b + c x^2 where 
    a - [n,m] matrix
    b - [n,1] vector
    c - [n,m] matrix
    '''
    print(f"a.shape = {a.shape}")
    print(f"b.shape = {b.shape}")
    print(f"c.shape = {c.shape}")

    #m is the dimension of x
    m = a.shape[1]

    #start with the guess x=0
    x = jnp.zeros( [m,] )

    for i in range(maxit):
        #Loop over each element of x and optimize the problem
        for j in range(m):
            #Let y be x but with y[j] = 0
            y = jnp.copy(x)
            y = y.at[j].set(0)
            #Set up a scalar optimization problem
            b2 = b + a @ y + c @ (y*y)
            xj, r = solve_scalar_problem( a[:,j], b2, c[:,j] )
            x = x.at[j].set(xj)

        #Print relative residual
        relative_res = r/jnp.linalg.norm(b)
        print(f"Iteration {i}: relative residual = {relative_res:.5e}")
    return x



def super_tensor_gmres( jac, f0, m, s_min, hessian_fn ):
    '''
    PURPOSE:
    The goal of this function is to modify our model of the objective function f to also contain partial 
    Hessian information along singular vectors with singular value below some threshold. 

    INPUT:
    jac - jacobian as a linear operator
    f0 - current value of objective function
    m - dimension of Krylov subspace
    
    OUTPUT:
    x - the predicted step for minimizing our objective function.
    '''

    #option for controlling the amount of information print to the screen
    verbose = True

    #Start with standard GMRES for constructing a Krylov subspace
    Q = []
    H = jnp.zeros((m+1, m))

    Q.append( f0 / jnp.linalg.norm( f0 ) )

    for k in range(m):
        qk = Q[k]
        v = jac(qk)

        for j in range(k+1):
            hj = jnp.dot(Q[j], v)
            H = H.at[j, k].set(hj)
            v = v - hj * Q[j]

        hk1 = jnp.linalg.norm(v)
        H = H.at[k+1, k].set(hk1)
        Q.append(v / hk1)

    # Form least squares problem: min ||beta*e1 - H y||
    Q = jnp.stack(Q, axis=1) # Shape: (n, m+1)

    #Project the constant vector onto the orthonormal basis
    b = Q.transpose() @ f0

    #Compute the singular value decomposition of our Hessenberg representation of the Jacobian.
    U, S, V = jnp.linalg.svd( H )

    #Determine how many "problematic" directions we have in our linear system.
    #Problematic means the singular value is smaller than s_min
    num_marginal = jnp.sum( S < s_min )

    print(f"We have {num_marginal} directions with sigma_n < {s_min}")
    print(f"Computing Hessian information for each of these directions...")

    #Store the problematic vectors in M and their Hessians in L
    L = jnp.zeros( [ m+1, num_marginal] )
    M = jnp.zeros( [ m,   num_marginal] )
    for i in range(num_marginal):
        #Store the problematic singular vector
        M = M.at[:,i].set( V[i-num_marginal,:] )

        #Evaluate the second derivative along this vector
        #lift the vector of M into the orginial vector space
        M_lift =  Q[:, :m] @ M[:,i]
        hessian_output = hessian_fn( M_lift )
        
        projection = Q.transpose() @ hessian_output
        if verbose:
            print(f"Relative magnitude {jnp.linalg.norm(projection) / jnp.linalg.norm(hessian_output):.3e} after projection...")

        L = L.at[:,i].set( projection )

        if verbose:
            print( f"sanity check: [{S[i-num_marginal]}, {jnp.linalg.norm( H @ M[:,i] )}]" )
    
    #Transform H on the right to move problematic vectors to the end.
    HV = H @ V

    #Perform a QR decomposition
    #Return the complete decomposition so we can do error estimation and debug
    W, R = jnp.linalg.qr(HV, mode="complete")

    # Rotate all vectors correspondingly
    b2 = W.transpose() @ b
    L2 = W.transpose() @ L

    #Take the sub-Jacobian that acts on the perpindicular space.
    #This matrix is of size (m+1)-by-(m-1)
    subR = R[:, :-num_marginal]
    a2 = R[:, -num_marginal:] #last columns

    #Take the last two elements of each vector and minimize the vector quadratic for xt
    #xt = minimize_vector_quadratic( a2[-(num_marginal+1):], b2[-(num_marginal+1):], L2[-(num_marginal+1):], verbose )

    maxit = 64
    xt = kaczmarz_iteration( a2[-(num_marginal+1):], b2[-(num_marginal+1):], 0.5*L2[-(num_marginal+1):], maxit )

    #Solve for the remaining elements by inverting the upper triangular system
    b_mod = b2 + a2 @ xt + 0.5 * L2 @ (xt*xt)
    y_perp, _, _, _ = jnp.linalg.lstsq( subR, -b_mod)

    #Stack s with y_perp
    #y = jnp.stack( (s,y_perp), axis=0 )
    y = jnp.zeros( [m,] )
    y = y.at[:-num_marginal].set(y_perp)
    y = y.at[-num_marginal:].set(xt)

    # after constructing y, don't forget the invert the svd rotation
    y = V.transpose() @ y

    #if verbose:
        #print(f"predict residual: {residual_prediction}")
        
        #Compute the residual in the Krylov basis before we did any linear algebra
        #r = b + H @ y + 0.5 * L @ (xt*xt)

        #print(f"actual  residual: {jnp.linalg.norm(r)}")
        #print(f"Rotated residual vector is { V.transpose() @ r }")
        #If my intuition for minimization is correct, then this vector should only be nonzero in last two elements 

    #Lift y back to Krylov subspace
    x = Q[:, :m] @ y

    return x

