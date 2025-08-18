
import jax
import jax.numpy as jnp

def minimize_vector_quadratic(a,b,c, verbose=False):
    '''
    Find x that minimizes the 2-norm of 

    b + a*x + 1/2*c*x^2,

    where {a,b,c} are vectors and x is a single scalar. 
    Squaring this expression gives a quartic function, and the derivative is a cubic in x.
    Solve that cubic and return the real x that minimizes the function.
    '''

    #Take all possible dot products
    aa = jnp.dot(a,a)
    ab = jnp.dot(a,b)
    ac = jnp.dot(a,c)
    bb = jnp.dot(b,b)
    bc = jnp.dot(b,c)
    cc = jnp.dot(c,c)

    #Construct and solve the cubic polynomial
    coeffs = jnp.array( [cc, 3*ac, 2*(bc+aa), 2*ab] )
    x = jnp.roots(coeffs)

    #Mask for "real enough" values
    imaginary_tolerance = 1e-10
    is_real = jnp.abs(jnp.imag(x)) < imaginary_tolerance
    x = x[is_real].real

    #Compute the squared residual of these x
    r = bb + 2*ab*x + (bc + aa)*x*x + ac*x*x*x + cc/4*x*x*x*x
    
    #Find the index of the minimum
    i = jnp.argmin(r)

    if verbose:
        print(f"\nminimizing vector quadratic with {a.shape} degrees of freedom...")
        print(f"x values are {x}")
        print(f"The corresponding 2-norms are {jnp.sqrt(r)}")
        print(f"Choosing x = {x[i]}...\n")

    #Return the minimizes of this function and the function value
    return x[i], jnp.sqrt(r[i])



def tensor_gmres( jac, f0, m, hessian_fn ):
    '''
    PURPOSE:
    Newton-GMRES starts to perform poorly when the Jacobian has singular values near zero. Rather than think carefully about how to remove these near-zeros,
    I would like to modify our approximation of the objective function with low-rank quadratic information. That is, we approximate our function f(z) not just with

    f(z) \approx f(0) + Jz,

    but with the additional quadratic term

    f(z) approx f(0) + Jz + 1/2 h (z dot hat{t})^2

    where hat{t} is the unit vector in the step direction from Newton-GMRES, and h is the Hessian information 
    
    h = (hat{t} cdot \partial) (hat{t} cdot \partial) f.

    This way, if J is not linearly responsive to some descent direction, the quadratic term can restore nonlinear sensitivity and prevent over-stepping in optimization.
    This function goes beyond a linear algebra routine because we must evaluate h somehow. 

    INPUT:
    jac - jacobian as a linear operator
    f0 - current value of objective function
    m - dimension of Krylov subspace
    
    OUTPUT:
    x - the predicted step for minimizing our objective function.
    '''

    #option for controlling the amount of information print to the screen
    verbose = True

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
    
    #This is the negative of the Newton step, but we only want the axis of interest
    y_gmres, _, _, _ = jnp.linalg.lstsq(H, b, rcond=None)
        
    if verbose:
        gmres_residual = jnp.linalg.norm( H @ y_gmres - b ) / jnp.linalg.norm( b )
        print(f"GMRES reached a relative residual of {gmres_residual:.3e}")

    #t will be the unit vector that points along x_gmres
    t = y_gmres / jnp.linalg.norm( y_gmres )

    #evaluate the Hessian function along t
    h = hessian_fn( Q[:, :m] @ t )

    #Porject the hessian action into the Krylov subsapce
    c = Q.transpose() @ h

    #See how much information we lose by projecting h into our Krylov basis
    if verbose:
        c_rel_mag = jnp.linalg.norm( c ) / jnp.linalg.norm( h )
        print(f"Projecting the Hessian into the Krylov subspace has relative magnitude {c_rel_mag:.3e}. 1 is perfect.")

    def rotation_for_t():
        # Construct a Householder reflection that maps a unit vector t -> e_n
        # where e_n is (0,0,0,0,...,0,1)
        # We can abuse the fact that t is a unit vector
        e = 0*t
        e = e.at[-1].set(1)
        v = t - e
        v = v/jnp.linalg.norm(v)
        return jnp.eye(m) - 2*jnp.outer(v,v)
    
    U = rotation_for_t()

    #if verbose:
    #    e = U @ t
    #    print( f"last elemnt of U @ t is {e[-1]}" )

    #Transform H on the right with this Householder transformation
    HU = H @ U

    #Perform a QR decomposition
    #Return the complete decomposition so we can do error estimation and debug
    V, R = jnp.linalg.qr(HU, mode="complete")

    # Rotate all vectors correspondingly
    b2 = V.transpose() @ b
    c2 = V.transpose() @ c

    #Take the sub-Jacobian that acts on the perpindicular space.
    #This matrix is of size (m+1)-by-(m-1)
    subR = R[:, :-1]
    a2 = R[:, -1] #remaining column

    #Take the last two elements of each vector and minimize the vector quadratic for xt
    xt, residual_prediction = minimize_vector_quadratic( a2[-2:], b2[-2:], c2[-2:], verbose )
    
    if verbose:
        print(f"Tensor gmres finds |dot(x,t)| = {jnp.abs(xt):.3e} compared to {jnp.linalg.norm(y_gmres):.3e} from GMRES alone.")

    #Solve for the remaining elements by inverting the upper triangular system
    b_mod = b2 + a2*xt + 0.5*c2*xt*xt
    y_perp, _, _, _ = jnp.linalg.lstsq( subR, -b_mod)

    #Stack s with y_perp
    #y = jnp.stack( (s,y_perp), axis=0 )
    y = 0*y_gmres
    y = y.at[:-1].set(y_perp)
    y = y.at[-1].set(xt)

    # after constructing y, don't forget the invert the Householder reflection that made all of this possible!
    # U is its own inverse
    y = U @ y

    if verbose:
        print(f"predict residual: {residual_prediction}")
        
        #Compute the residual in the Krylov basis before we did any linear algebra
        r = b + H @ y + 0.5 * c * jnp.dot(t,y)**2

        print(f"actual  residual: {jnp.linalg.norm(r)}")
        #print(f"Rotated residual vector is { V.transpose() @ r }")
        #If my intuition for minimization is correct, then this vector should only be nonzero in last two elements 

    #Lift y back to Krylov subspace
    x = Q[:, :m] @ y

    return x

