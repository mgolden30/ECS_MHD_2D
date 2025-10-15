'''
NOTES:

October 8th, 2025
I'm doing a major rewrite of this file in an effort to standardize these functions to some extent
and give more power to the user in selecting and defining their own time integration.
In particular, I want to enforce these conditions:
    1. These functions should be abstract integration routines.
    2. The first two arguments should always be the initial condition f and time t.
'''


import jax
import jax.flatten_util
import jax.numpy as jnp

def rk4(f, t, steps, v_fn):
    """
    Classic fourth order Runge-Kutta
    
    Parameters
    ----------
    f : array
        initial condition f(0)
    t : float
        integration time
    steps : int
        number of timesteps
    v_fn : callable
        Computes the time derivative of the state
        
    Returns
    -------
    f : array
        The numerical approximation of f(t)
    """
    h = t/steps
    def update_f(_, f):
        k1 = h * v_fn(f)
        k2 = h * v_fn(f + k1/2)
        k3 = h * v_fn(f + k2/2)
        k4 = h * v_fn(f + k3)
        return f + k1/6 + k2/3 + k3/3 + k4/6
    return jax.lax.fori_loop( 0, steps, update_f, f)

def lawson_rk4(f, t, steps, v_fn, L_diag, mask=None):
    """
    Classic fourth order Runge-Kutta applied to Lawson integration as proposed in 
    "Generalized Runge-Kutta processes for stable systems with large Lipschitz constants" by J. Lawson 1967.
    
    Parameters
    ----------
    f : array
        initial condition f(0)
    t : float
        integration time
    steps : int
        number of timesteps
    v_fn : callable
        Computes the (nonlinear) time derivative of the state
    L_diag : array
        The linear dynamics to handle implicitly. I assume these dynamics are diagonal, so L_diag is the same shape as f.
        
    Returns
    -------
    f : array
        The numerical approximation of f(t)
    """
    h = t/steps
    e = jnp.exp( h/2 * L_diag )
    if mask is not None:
        e = e*mask
    def update_f(_, f):
        k1 = h * v_fn(f)
        f = e*f; k1 = e*k1;
        k2 = h * v_fn(f + k1/2)
        k3 = h * v_fn(f + k2/2)
        f = e*f; k1 = e*k1; k2 = e*k2; k3 = e*k3;        
        k4 = h * v_fn(f + k3)
        return f + k1/6 + k2/3 + k3/3 + k4/6
    return jax.lax.fori_loop( 0, steps, update_f, f)

def lawson_rk6(f, t, steps, v_fn, L_diag, mask=None):
    """
    Sixth order Runge-Kutta applied to Lawson integration 

    Parameters
    ----------
    f : array
        initial condition f(0)
    t : float
        integration time
    steps : int
        number of timesteps
    v_fn : callable
        Computes the (nonlinear) time derivative of the state
    L_diag : array
        The linear dynamics to handle implicitly. I assume these dynamics are diagonal, so L_diag is the same shape as f.
        
    Returns
    -------
    f : array
        The numerical approximation of f(t)
    """
    h = t/steps
    e = jnp.exp( h/6 * L_diag )
    if mask is not None:
        e = e*mask
    def update_f(_, f):
        k1 = h * v_fn(f)
        f = e*f; k1 = e*k1;
        k2 = h * v_fn(f + k1/6)
        k3 = h * v_fn(f + k1/12 + k2/12);
        f = e*f; k1 = e*k1; k2 = e*k2; k3 = e*k3;        
        k4 = h * v_fn(f - k2*4/33 + k3*5/11);
        f = e*f; k1 = e*k1; k2 = e*k2; k3 = e*k3; k4 = e*k4
        k5 = h * v_fn(f - k1/4 - k2*29/44 + k3*31/22);
        f = e*f; k1 = e*k1; k2 = e*k2; k3 = e*k3; k4 = e*k4; k5 = e*k5
        k6 = h * v_fn(f + k1*3/11 + k2*8/33 - k3*4/11 + k4/11 + k5*14/33);
        f = e*f; k1 = e*k1; k2 = e*k2; k3 = e*k3; k4 = e*k4; k5 = e*k5; k6 = e*k6
        k7 = h * v_fn(f - k1*17/48 - k2*5/12 + k3 + k4 - k5*13/12 + k6*11/16);
        f = e*f; k1 = e*k1; k2 = e*k2; k3 = e*k3; k4 = e*k4; k5 = e*k5; k6 = e*k6; k7 = e*k7
        k8 = h * v_fn(f + k1*20/39 + k2*12/39 - k3*31/39 - k4/39 + k5*34/39 - k6*11/39 + k7*16/39);             
        return f + 13/200*(k1+k8) + 4/25*(k3+k7) + 11/40*(k4+k6)
    return jax.lax.fori_loop( 0, steps, update_f, f)

def tdrk4(f, t, steps, v_fn):
    '''
    Perform forward time evolution with Two Derivative RK4 (TDRK4).
    This integration scheme requires evaulation of the second time derivative,
    which we will achieve with autodiff.
    '''
    h = t/steps
    #define the Jacobian-vector operator with autodiff
    jac = lambda f, k: jax.jvp( v_fn, primals=(f,), tangents=(k,) )[1]
    def update_f(_, f):
        #Stage 1
        k1 = h * v_fn(f)
        a1 = h * jac(f, k1)
        #Stage 2
        f_temp = f + k1/2 + a1/8
        k2 = h * v_fn(f_temp)
        a2 = h * jac(f_temp, k2)
        #quadrature
        f = f + k1 + a1/6 + a2/3
        return f
    #Apply to update 
    f = jax.lax.fori_loop( 0, steps, update_f, f)
    return f

def lawson_rk43( x, vel_fn, diag_L, t, h=1e-2, atol=1e-4, max_steps_per_checkpoint=1024, checkpoints=1):
    '''
    Adaptively integrate with Lawson RK4. An embedded third order scheme is used to determine a good timestep.
    
    This does require repeated evaluation of the exponential of dissipation, which may or may not be acceptable
    for your use case.

    Parameters
    ----------
    x : ndarray
        Initial condition
    vel_fn : callable
        A function that responds to vel_fn(x) and returns the state velocity
    diag_L: ndarray
        The diagonal of a linear operator L. Should be the same size as x so that
        L @ x = diag_L * x in python. This restricts us to diagonal operators, which is
        still very useful for our doubly periodic domain. It is trivial to generalize this code
        to non-diagonal L, but we will not consider it.
    t : float
        Desired integration time
    h : float (optional)
        Initial guess of timestep.
    atol: float (optional)
        Absolution TOLerance (ATOL). Accept the step if norm(error) < atol.
    max_steps_per_checkpoint: int (optional)
        Hardcoding a limit per checkpoint has two benefits.
            1) Prevents the code from stalling as h->0
            2) Allows reverse mode differentiation for things like adjoint looping.
    checkpoints: int (optional)
        Reverse mode differentiation exhausts memory very quickly since every integration step
        is stored. Force integration to checkpoint into evenly spaced segments.

    Returns
    -------
    x : ndarray
        The final state.
    info : dictionary
        info is a dictionary containing useful diagnostic information.
        "completed": a Boolean flag. Did we hit the desired integration time?
        "s": time of integration. If completed is True, then s == 1. If completed is False, 
             then info["s"] contains the pseudotime we did integrate to before hitting max_steps. 
        "accepted": number of integration steps accepted during integration
        "rejected": number of integration steps rejected during integration
        "fevals": number of times we evaluated the nonlinear part of the time derivative.


    Notes
    -------
    Adaptive timestep integration of the system of ODEs 
    
    dx/dt = f(x) + Lx, 

    where f(x) is a nonlinear function and L is a stiff linear operator.
    We assume in this implementation that L is diagonal, but the method described here
    works for all L.

    Any Runge-Kutta method can be used to generte a Lawson integration scheme.
    
    A fourth order EARK4 scheme must solve 21 nontrivial order conditions. I found a very pretty
    modification of traditional RK4 that accomplishes this. It can be embedded with a third order 
    scheme for easy error estimation and stepsize control. 
    0   |                          
    1/2 |  1/2                      
    1/2 |  0    1/2                 
    1   |  0    0    1              
    1   |  1/6  1/3  1/3  1/6      
    _________________________________
    4th |  1/6  1/3  1/3  1/6   0   
    3rd |  1/6  1/3  1/3  1/3  -1/6 
    _________________________________
    err |  0    0    0   -1/6   1/6 


    Autodiff wisdom
    -------
    To make reverse-mode autodiff stable, I made two changes. I believe both are important.
    1) stop gradient calculations of h. That is, autodiff sees the time grid as a constant.
    2) reformulate the ODE from dx/dt = f(x) to dx/ds = t*f(x)
    
    Bug fixes
    ----------
    1) Added nan-checking for err. If err became nan from a failed timestep, h would become nan permenantly.
       Now if err becomes nan, we try h/2 next timestep.
    2) I was updating h for the next timestep BEFORE updating s. The order is correct now.
    3) requiring s == 1.0 for integration to be "completed" is unstable. I now just check that jnp.abs(s-1.0) < threshold.
    '''

    #Monitor the timesteps we take
    hs = jnp.zeros([max_steps_per_checkpoint*checkpoints,])

    #Define modified dynamics: dx/ds = t*f(x)
    mod_L = t * diag_L
    mod_vel_fn = lambda x: t * vel_fn(x)

    # Create a dict to specify the state along the while loop
    # x - current value x(tau)
    # s - pseudotime ranging from 0 to 1
    # h - timestep
    # k1 - velocity vector that we save between integration steps
    # fevals - number of function evaluations
    # accepted - number of accepted steps
    # rejected - number of rejected steps
    # hs - a complete history of timesteps attempted and used by our integration.
    #loop_state = {"x": x, "s": 0.0, "h": h, "k1": mod_vel_fn(x), "fevals": 1, "accepted": 0, "rejected": 0, "hs": hs}
    loop_state = (x, 0.0, h, mod_vel_fn(x), 1, 0, 0, hs)

    def do_step(loop_state):
        x0, s, h, k1, fevals, accepted, rejected, hs = loop_state
        #x0 = loop_state["x"]
        #h  = loop_state["h"]

        #Compute an exponential of our diagonal matrix
        e = jnp.exp( mod_L * h/2 )
        
        #update with exponential twiddle
        x  = e*x0
        k1 = e*k1

        #Evaluate next two stages
        k2 = mod_vel_fn(x + h*k1/2)
        k3 = mod_vel_fn(x + h*k2/2)

        #update with exponential twiddle
        x  = e*x
        k1 = e*k1
        k2 = e*k2
        k3 = e*k3

        #Evaluate next two stages
        k4 = mod_vel_fn(x + h*k3)
        xf = x + h*(k1/6 + k2/3 + k3/3 + k4/6 )
        k5 = mod_vel_fn(xf)

        #Regardless of loop logic, we did four function evaluations 
        fevals = fevals + 4

        #estimate the error from this step 
        err = h*(k5-k4)/6
        #err = jax.numpy.linalg.norm(err)
        err = jnp.fft.irfft2(err) #Evaluate pointwise error in real space, not Fourier
        err = jnp.max(jnp.abs(err)) #A pointwise maximum error seems natural

        #Record the current h we used this step
        hs = hs.at[accepted + rejected].set(h)

        #Determine if this step is accepted or rejected.
        xf, k1, s, accepted, rejected = jax.lax.cond( err < atol, lambda _: (xf, k5, s + h, accepted + 1, rejected), lambda _: (x0, k1, s, accepted, rejected + 1), None )

        #Determine the next timestep. Now that s is updated and we don't need the current h value any more.
        adapt = lambda h: h * 0.9 * (atol / err)**(1/4) #Run this if err is finite
        crash = lambda h: h/2 #Run this if the sim crashed: err is nan
        h = jax.lax.cond( jnp.isnan(err), crash, adapt, operand=h )

        def clip_stepsize(h, s):
            overshoot = s + h - 1
            return h - jax.nn.relu(overshoot)
        h = clip_stepsize(h, s)
        h = jax.lax.stop_gradient(h) 

        return  (xf, s, h, k1, fevals, accepted, rejected, hs)

    #Determine when we completed integration.
    complete_fn = lambda s: jnp.abs(s - 1.0) < 1e-10

    def scan_fn(loop_state, _): #ignore carry index
        s = loop_state[1]
        new_state = jax.lax.cond( complete_fn(s), lambda x: x, lambda x: do_step(x), operand=loop_state )
        return new_state, None

    inner_loop = jax.checkpoint( lambda loop_state, _: jax.lax.scan(scan_fn, loop_state, xs=None, length=max_steps_per_checkpoint) )
    loop_state, _ = jax.lax.scan(inner_loop, loop_state, xs=None, length=checkpoints)

    #unpack the state
    x, s, h, _, fevals, accepted, rejected, hs = loop_state

    #Run information that should be of interest to the user
    info = {"completed": complete_fn(s),
            "s": s, #If not completed, this will tell you how far along you integrated. 
            "accepted": accepted,
            "rejected": rejected,
            "fevals": fevals,
            "hs": hs
           }

    return x, info

















if __name__ == "__main__":
    """
    If we run this file, do some benchmarking and testing of the adaptive integrators.
    """

    import time
    import mhd_jax as mhd_jax
    
    precision = jnp.float64
    # If you want double precision, change JAX defaults
    if (precision == jnp.float64):
        jax.config.update("jax_enable_x64", True)
    
    n   = 128 #spatial resolution
    nu  = 1/40 #fluid dissipation
    eta = 1/40 #magnetic dissipation
    b0  = [0.0, 0.1] # Mean magnetic field

    # Construct a dictionary for grid information
    param_dict = mhd_jax.construct_domain(n, precision)

    # Get grids
    x = param_dict['x']
    y = param_dict['y']
    forcing = -4*jnp.cos(4*y) #Kolmogorov forcing

    # Append the extra system information to param_dict
    param_dict.update( {'nu': nu, 'eta': eta, 'b0': b0, 'forcing': forcing} )

    # Initial data
    f = jnp.zeros([2, n, n], dtype=precision)
    f = f.at[0, :, :].set( jnp.cos(4*x-0.1)*jnp.sin(x+y-1.2) - jnp.sin(3*x-1)*jnp.cos(y-1) + 2*jnp.cos(2*x-1))
    f = f.at[1, :, :].set( jnp.cos(3*x+2.1)*jnp.sin(y+3.5) - jnp.cos(1-x) + jnp.sin(x + 5*y - 1 ) )

    #Construct our dissipation operator
    coeffs = jnp.array([nu, eta], dtype=precision)
    coeffs = jnp.reshape(coeffs, [2,1,1] )
    k_sq = param_dict['kx']**2 + param_dict['ky']**2
    diag_L = - coeffs * k_sq #This is the operator

    #Desired time to integrate
    t = 2.0
    atol = 1e-2 #acceptable error per step

    print("Testing adaptive integration on MHD...\n")







    print("\nRunning without JIT...")

    #Capture param_dict
    def vel_fn(f):
        return mhd_jax.state_vel(f, param_dict, include_dissipation=False )

    #wrap eark43 to take a dictionary and do FFTs 
    def wrapper( input_dict ):
        f = input_dict["f"]
        t = input_dict["t"]
        f = jnp.fft.rfft2(f) * param_dict['mask']
        f, info = eark43( f, vel_fn, diag_L, t, h=1e-2, atol=atol, max_steps_per_checkpoint=64, checkpoints=32)
        f = jnp.fft.irfft2(f)
        return f, info

    input_dict = {"f":f, "t":t}

    start = time.time()
    xf, info = wrapper(input_dict)
    stop = time.time()

    print( f"Function returned after {stop - start:.6f} seconds.\n")







    print("\nRunning with JIT...")

    integrate = jax.jit( wrapper )
    _ = integrate(input_dict)

    start = time.time()
    xf, info = integrate( input_dict )
    stop = time.time()

    print( f"Function returned after {stop - start:.6f} seconds.\n")
    print(info)




    print("\nUsing autodiff to evaluate JVP")

    input_dict = {"f": f, "t": t}

    def primal_fn(input_dict):
        y, _ = wrapper(input_dict)
        return y

    jacobian = jax.jit( lambda v: jax.jvp( primal_fn, primals=(input_dict,), tangents=(v,) )[1] )
    _ = jacobian(input_dict)

    #Define a tangent vector of intereset
    g = jnp.zeros([2, n, n], dtype=precision)
    g = g.at[0, :, :].set( jnp.cos(x) )
    g = g.at[1, :, :].set( jnp.cos(y) )

    #Tangent dict
    tang = {"f": g, "t": 1.0}    

    start = time.time()
    Jv = jacobian(tang)
    stop = time.time()
    
    print( f"Function returned after {stop-start:.6f} seconds." )
    print( f"mean(abs(answer)) = {jnp.mean(jnp.abs(Jv))}")




    print("\nUsing autodiff to evaluate VJP (adjoint)")

    _, jac_t = jax.vjp( primal_fn, input_dict, has_aux=False )

    jac_t = jax.jit(jac_t)

    _ = jac_t(g)

    start = time.time()
    uJ = jac_t(g)[0]
    stop = time.time()
    print( f"Function returned after {stop-start:.6f} seconds." )
    data = uJ["t"]
    print( f"temporal gradient = {data}")
    data = uJ["f"][0,3,2]
    print( f"field gradient = {data}")


    print("\nChecking that (uJ,v) = (u,Jv) to reasonable precision...")

    dot1 = jnp.sum( g * Jv )

    to_vec = lambda d: jax.flatten_util.ravel_pytree(d)[0]
    dot2 = jnp.sum( to_vec(uJ) * to_vec(tang) )

    print(f"(u,Jv) = {dot1}")
    print(f"(uJ,v) = {dot2}")