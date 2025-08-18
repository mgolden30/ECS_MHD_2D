'''
Written June 19th 2025 by Matthew Golden

PURPOSE:
I want a general purpose JAX code for Newton-Krylov iteration.

Version History:

'''

import jax
import jax.flatten_util
import jax.numpy as jnp

import lib.linalg as linalg

def newton_krylov( F_dict, x, maxit, inner, damp, precond, dealias ):
    '''
    F - a function handle to a function we want to perform root finding for
        assume F maps dictionaries to dictionaries
    x - our guess at F(x) = 0
        assume x is a dictionary
    '''

    #Jax comes with a utility to turn a dictionaries into a single vector.
    #Let's use it liberally.
    _, unravel_fn_x = jax.flatten_util.ravel_pytree(x)

    #Define a vector -> vector function using this utility so that we can easilly do Krylov iteration 
    F = lambda x: jax.flatten_util.ravel_pytree(F_dict(unravel_fn_x(x)))[0]

    #Define the action of the Jacobian
    jac  = jax.jit( lambda primal, tangent: jax.jvp( F, (primal,), (tangent,))[1] )
    
    #Define the action of the Jacobian adjoint
    jacT = jax.jit( lambda primal, cotangent: jax.vjp( F, (primal))[1](cotangent)[0] )


    for i in range(maxit):
        #Flatten state to a vector
        x_vec, _ = jax.flatten_util.ravel_pytree(x)

        #Evaluate the vector -> vector function 
        f = F(x_vec)

        #print how big f is at this iteration.
        print( f"Iteration {i}: |f|_2 = {jnp.linalg.norm(f)}, max|f| = {jnp.max(jnp.abs(f))}" )

        #Get the step with a Krylov method.
        m = f.size
        n = x_vec.size

        A   = lambda v: jac(x_vec, v)
        A_t = lambda v: jacT(x_vec, v)

        dx = linalg.adjoint_GMRES( A, A_t, f, m, n, inner, precond)

        #Update state
        x_vec = x_vec - damp*dx
        x = unravel_fn_x(x_vec)
        x = dealias(x)
    
    return x