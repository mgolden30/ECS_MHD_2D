'''
The default implementation of jax.scipy.sparse.linalg.gmres is flawed and gives memory bugs for no reason.
Here we implement our own GMRES routines.
'''

import jax.numpy as jnp

from scipy.io import savemat

def gmres(A, b, m, Q0, tol=1e-8):
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
    s_min = 0.5 #smallest singular value we are comfortable inverting
    inv_s = inv_s.at[ s < s_min ].set(1)

    b2 = b2 * inv_s
    y  = Vh.T @ b2

    x = Qmat[:, :m] @ y
    return x



def block_gmres(A, b, m, B, tol=1e-8):
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
    savemat("bgmres.mat", {"H": H, "Q": Q} )

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