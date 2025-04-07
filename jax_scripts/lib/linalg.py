'''
The default implementation of jax.scipy.sparse.linalg.gmres is flawed and gives memory bugs for no reason.
Here we implement our own GMRES routines.
'''

import jax.numpy as jnp

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


def ChatGPT_gmres(A, b, m, tol=1e-8):
    n = b.shape[0]
    Q = []
    H = jnp.zeros((m+1, m))

    x0 = jnp.zeros_like(b)
    r0 = b - A(x0)
    beta = jnp.linalg.norm(r0)
    Q.append(r0 / beta)

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
    e1 = jnp.zeros(m+1).at[0].set(beta)
    y, _, _, _ = jnp.linalg.lstsq(H, e1, rcond=None)

    x = x0 + Qmat[:, :m] @ y
    return x