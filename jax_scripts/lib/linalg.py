'''
The default implementation of jax.scipy.sparse.linalg.gmres is flawed and gives memory bugs for no reason.
Here we implement our own GMRES routines.
'''

import jax
import jax.numpy as jnp

from scipy.io import savemat



def gmres(A, b, m, Q0, s_min, tol=1e-8):
    '''
    PURPOSE:
    Solve the linear system Ax=b by constructing a Krylov subspace.
    
    INPUT:
    A - a linear operator
    b - right hand side vector
    m - dimension of Krylov subspace
    Q0 - initial vector of Q
    '''

    Q = []
    H = jnp.zeros((m+1, m))
    Q.append( Q0 / jnp.linalg.norm(Q0) )

    for k in range(m):
        qk = Q[k]
        v = A(qk)
        for j in range(k+1):
            hj = jnp.dot(Q[j], v)
            H = H.at[j, k].set(hj)
            v = v - hj * Q[j]
        hk1 = jnp.linalg.norm(v)
        H = H.at[k+1, k].set(hk1)
        if hk1 < tol:
            break
        Q.append(v / hk1)

    # Form least squares problem: min ||beta*e1 - H y||
    Qmat = jnp.stack(Q, axis=1)  # Shape: (n, m+1)

    #Project b onto the orthonormal basis
    b2 = Qmat.transpose() @ b
    
    #e1 = jnp.zeros(m+1).at[0].set(beta)
    #y, _, _, _ = jnp.linalg.lstsq(H, b2, rcond=None)

    U, s, Vh = jnp.linalg.svd(H, full_matrices=False)

    b2 = U.T @ b2

    inv_s = 1 / s

    #NEVER INCREASE THE SIZE OF A STABLE DIRECTION
    inv_s = inv_s.at[ s < s_min ].set(1)

    b2 = b2 * inv_s
    y  = Vh.T @ b2

    x = Qmat[:, :m] @ y
    return x



def block_gmres(A, b, m, B, tol=1e-8, iteration=0):
    '''
    PURPOSE:
    Typical GMRES solves the linear system Ax=b by constructing a Krylov subspace
    K = {v, Av, A^2v, A^3v, ...}. 
    This is terrible, because you need to wait for Av to finish evaluating before starting A^2v.
    This subspace generation is too sequential and does not abuse the massively parallel computers 
    we have in the modern world.

    block_gmres will instead iterate an initial block of vectors 
    K = {B, AB, A^2B, ...} where B is the block. K is taken to be the span of the columns of this set.
    B is unrelated to b, although you might want B to contain b as a column.
        
    INPUT:
    A - a linear operator
    b - right hand side vector
    m - number of times to multiply our block by A
    B - block of vectors
    '''

    #block size
    n = B.shape[0]
    s = B.shape[1]

    Q = jnp.zeros((n, s*(m+1)) )
    H = jnp.zeros((s*(m+1), s*m))

    #Orthonormalize the block before we begin function evaluation
    B, _ = jnp.linalg.qr(B, mode='reduced')

    #Use the orthonormal block to start off our orthonormal basis
    Q = Q.at[:, 0:s].set(B)

    #Apply the operator m times
    for k in range(m):
        #Apply the linear operator to the block
        C = A(B)

        #Loop over the columns of C
        for i in range(s):
            #i is the index of the column relative to C
            #The corresponding index of Q is needed. Call this a.
            a = k*s + i
            for j in range(a+s):
                #Project current vector
                h = jnp.dot(Q[:,j], C[:,i])
                
                #Update the Hessenberg matrix
                H = H.at[j, a].set(h)
                C = C.at[:, i].set(C[:,i] - h * Q[:,j] )

            #Check if we have a new vector to add
            h = jnp.linalg.norm(C[:,i])
            if h < tol:
                print("Oh no")
                break

            H = H.at[a+s,a].set(h)
            Q = Q.at[:,a+s].set(C[:,i]/h)
            C = C.at[:,i].set(C[:,i]/h)
        
            #Reorthogonalize
            #I found this MANDATORY for numerical stability for subspaces with more than ~50 vectors
            for j in range(a+s):
                h = jnp.dot(Q[:,j], Q[:,a+s])
                Q = Q.at[:,a+s].set(Q[:,a+s] - h * Q[:,j])
        B = C
    #For debugging
    savemat(f"debug/bgmres_{iteration}.mat", {"H": H, "Q": Q} )

    #Project b onto the orthonormal basis
    b2 = Q.T @ b
    
    U, s, Vh = jnp.linalg.svd(H, full_matrices=False)

    b2 = U.T @ b2
    inv_s = 1 / s

    #NEVER INCREASE THE SIZE OF A STABLE DIRECTION
    s_min = 1.0 #smallest singular value we are comfortable inverting
    inv_s = inv_s.at[ s < s_min ].set(1)

    b2 = b2 * inv_s
    y  = Vh.T @ b2

    x = Q[:, :y.shape[0]] @ y
    return x



def adjoint_GMRES( A, A_t, b, m, n, inner, precond):
    '''
    PURPOSE:
    Traditional GMRES requires the matrix A to be square so that power iteration
    can be applied. This is stupid. Why should A be square? Even if A is square, not all R^n are equivalent physically.
    What physical meaning does power iteration actually have in such contexts?

    In light of this thinking, consider the linear map A : X -> Y, where X and Y are vector spaces.
    We want to solve the linear system Ax=b, where x is in X and b is in Y. We build an orthonormal basis of X and Y
    with Gram-Schmidt iteration by applying A and the adjoint of A.

    INPUT:
    A - linear operator
    A_t - adjoint (transpose) linear operator
    b - right hand side (solve Ax=b)
    m,n - Assume A is an m-by-n matrix. These are just matrix dimensions
    inner - how many inner iterations to do. an inner iteration consists of both an evaluation of A and A_t.
    precond - preconditioner

    OUTPUT:
    x - the approximate solution to Ax=b
    '''

    B = jnp.zeros((inner+1, inner))
    U = jnp.zeros((m, inner+1))
    V = jnp.zeros((n, inner))

    #Apply preconditioner to b
    b = precond( b, "notrans" )

    #Generate our orthonormal basis vectors with b
    U = U.at[:,0].set( b / jnp.linalg.norm(b) )

    #Power iteration
    for i in range(inner):
        #Apply (MA)^T
        Au = precond(U[:,i], "trans")
        Au = A_t(Au)

        #Orthogonalize with respect to previous V
        for j in range(i):
            B = B.at[i,j].set(jnp.dot( V[:,j], Au ))
            Au = Au - B[i,j]*V[:,j]

        #Define new vector of V
        B = B.at[i,i].set(jnp.linalg.norm(Au))
        V = V.at[:,i].set( Au / B[i,i] )

        #Apply MA
        Av = A(V[:,i])
        Av = precond(Av, "notrans")

        #Orthogonalize    
        for j in range(i+1):
            B  = B.at[j,i].set(jnp.dot( U[:,j], Av ))
            Av = Av - B[j,i]*U[:,j]
    
        B = B.at[i+1,i].set(jnp.linalg.norm(Av))
        U = U.at[:,i+1].set(Av / B[i+1,i])

    #You have constructed an orthonormal basis
    b2 = jnp.zeros(inner+1)
    b2 = b2.at[0].set(jnp.linalg.norm(b))

    #Solve least squares problem
    x2, _, _, _ = jnp.linalg.lstsq(B, b2, rcond=None)
    
    #Rotate to full space
    x = V @ x2

    return x