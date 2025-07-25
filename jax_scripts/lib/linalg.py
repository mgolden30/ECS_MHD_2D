'''
The default implementation of jax.scipy.sparse.linalg.gmres is flawed and gives memory bugs for no reason.
Here we implement our own GMRES routines.
'''

import jax
import jax.numpy as jnp

from scipy.io import savemat



def gmres(A, b, m, Q0, s_min, tol=1e-8, preconditioner_list=[], output_index=0 ):
    '''
    PURPOSE:
    Solve the linear system Ax=b by constructing a Krylov subspace.
    
    INPUT:
    A - a linear operator
    b - right hand side vector
    m - dimension of Krylov subspace
    Q0 - initial vector of Q

    preconditioner_list - an arbitrary length list of preconditioners (M1, M2, ...) to apply to the linear system.
    GMRES is then applied to the system  Mn*...*M2*M1*A*x = Mn*...*M2*M1*b
    Each M is a function handle so it evaluates via M(v)
    '''

    Q = []
    H = jnp.zeros((m+1, m))

    #Apply preconditioners to both b and Q0
    for M in preconditioner_list:
        b  = M(b)
        Q0 = M(Q0)

    Q.append( Q0 / jnp.linalg.norm(Q0) )

    for k in range(m):
        qk = Q[k]
        v = A(qk)
        for M in preconditioner_list:
            v = M(v)

        for j in range(k+1):
            hj = jnp.dot(Q[j], v)
            H = H.at[j, k].set(hj)
            v = v - hj * Q[j]

        #Reorthogonalize a couple of times
        for _ in range(2):
            for j in range(k+1):
                hj = jnp.dot(Q[j], v)
                v = v - hj * Q[j]


        hk1 = jnp.linalg.norm(v)
        H = H.at[k+1, k].set(hk1)
        Q.append(v / hk1)

    # Form least squares problem: min ||beta*e1 - H y||
    Qmat = jnp.stack(Q, axis=1) # Shape: (n, m+1)

    #Project b onto the orthonormal basis
    b2 = Qmat.transpose() @ b
    
    #e1 = jnp.zeros(m+1).at[0].set(beta)
    #y, _, _, _ = jnp.linalg.lstsq(H, b2, rcond=None)

    filename = f"gmres_debug_{output_index}.mat"
    savemat(filename, {"H": H, "b": b2, "f": b, "Q": Q})

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



def adjoint_GMRES( A, A_t, b, m, n, inner, outer=1, preconditioner_list=[]):
    '''
    PURPOSE:
    Minimize the 2 norm of Ax-b where x is constrained to the subspace span( A^T b, (A^T A) A^T b, ..., (A^T A)^n A^T b )
    This differs from typical GMRES where A is assumed square. Here A can be non-square, but it requires the implementation
    of the transpose.

    INPUT:
    A - linear operator
    A_t - adjoint (transpose) linear operator
    b - right hand side
    m,n - A is a m-by-n matrix
    inner - how many inner iterations to do. an inner iteration consists of both an evaluation of A and A_t.
    outer - if memory cannot support good residuals by inner iteration, we have no choice but to restart.
    preconditioner_list=[M1,M2,...] a list of function handles to precondition with.
    
    OUTPUT:
    x - the approximate solution to Ax=b
    '''

    #The three matrices we construct every inner iteration
    B = jnp.zeros((inner+1, inner))
    U = jnp.zeros((m, inner+1))
    V = jnp.zeros((n, inner))

    #The approximate solution to Ax=b
    x = jnp.zeros((n,))

    #Compute the current remaining right hand side
    #Call this c to distinguish from b and update it every outer iteration.
    c = jnp.copy(b)
 
    #Apply preconditioners to c
    for M in preconditioner_list:
        c = M(c, "no_trans")

    #Compute the norm of c initially so we can print the relative residual each outer iteration
    c0 = jnp.copy(c)
    norm_c0 = jnp.linalg.norm(c0)

    #print out the rel res each outer iteration
    verbose = True

    for outer_iteration in range(outer):
        #Store the norm so we can solve Ax=\hat{c}.
        #This should maintain numerical stability as norm(c) -> 0
        norm_c = jnp.linalg.norm(c)

        #Generate our orthonormal basis vectors with c
        U = U.at[:,0].set( c / norm_c )

        #Start power iteration
        for i in range(inner):

            #Apply (MA)^T
            Au = U[:,i]
            for M in preconditioner_list:
                Au = M(Au, "trans")
            Au = A_t(Au)

            #Orthogonalize with respect to previous V
            for j in range(i):
                B = B.at[i,j].set(jnp.dot( V[:,j], Au ))
                Au = Au - B[i,j]*V[:,j]
            
            #Reorthogonalize to ensure numerical stability
            for _ in range(2):
                for j in range(i):
                    Bij = jnp.dot( V[:,j], Au )
                    Au = Au - Bij*V[:,j]
                
            #Define new vector of V
            B = B.at[i,i].set(jnp.linalg.norm(Au))
            V = V.at[:,i].set( Au / B[i,i] )

            #Apply MA
            Av = A(V[:,i])
            for M in preconditioner_list:
                Av = M(Av, "no_trans")

            #Orthogonalize    
            for j in range(i+1):
                B  = B.at[j,i].set(jnp.dot( U[:,j], Av ))
                Av = Av - B[j,i]*U[:,j]
        
            #Reorthogonalize to ensure numerical stability
            for _ in range(2):
                for j in range(i+1):
                    Bji  = jnp.dot( U[:,j], Av )
                    Av = Av - Bji*U[:,j]

            B = B.at[i+1,i].set(jnp.linalg.norm(Av))
            U = U.at[:,i+1].set(Av / B[i+1,i])

        #Project c into the basis U
        #Since we used c to generate U, it is just e1
        e1 = jnp.zeros(inner+1)
        e1 = e1.at[0].set(1)


        #Solve least squares problem
        #TODO, just do Given's rotations on the three diagonals in a smart way.
        y, _, _, _ = jnp.linalg.lstsq(B, e1, rcond=None)
        
        # (1) Lift y into the full space (multiply by V) 
        # (2) Rescale by c_norm since we solved for e1
        dx = V @ (y*norm_c)

        #Add this to our guess of x
        x = x + dx

        #Update c. Do a matrix evaluation. Eh, it's cheap enough.
        Ax = A(x)
        for M in preconditioner_list:
                Ax = M(Ax, "no_trans")
        c = c0 - Ax

        if verbose:
            rel_res = jnp.linalg.norm(c) / norm_c0
            print(f"outer iteration {outer_iteration}: relative residual = {rel_res:.6e}")
    
        #savemat("debug_adjoint_GMRES.mat", {"U": U, "V": V, "B": B})

    return x





