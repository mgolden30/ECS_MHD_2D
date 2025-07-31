import jax
import jax.numpy as jnp


def runge_kutta_32( x, vel_fn, t, h=1e-2, atol=1e-4):
    '''
    Do explicit time integration with RK3(2)
    '''

    #Create a dict to specify the state along the while loop
    #x - current value x(tau)
    #tau - time integrated so far.
    #h - timestep
    #k1 - velocity vector that we save between integration steps
    #fevals - number of function evaluations
    loop_state = {"x": x, "tau": 0.0, "h": h, "k1": vel_fn(x), "fevals": 1, "accepted": 0, "rejected": 0}

    # Define loop condition: takes the loop state and returns a boolean
    def cond_fun(loop_state):
        return loop_state["tau"] < t

    # Define loop body: takes the loop state and returns a new loop state
    def body_fun(loop_state):
        x = loop_state["x"]
        k1= loop_state["k1"]
        h = loop_state["h"]

        #Normally I would define k = h*f(x), but h is changing each step
        k2 = vel_fn(x + h*k1/2)
        k3 = vel_fn(x + h*k2*3/4)
        xf = x + h*(2/9*k1 + 3/9*k2 + 4/9*k3 )
        k4 = vel_fn(xf)

        #Regardless of loop logic, we did three function evaluations 
        fevals = loop_state["fevals"] + 3

        #Use all four velocity evaluations to estimate the error from this step 
        err = h*( k1*(3/10-2/9) + k2*(6/25-3/9) + k3*(8/25-4/9) + k4*(7/50-0) )
        err = jax.numpy.linalg.norm(err)

        #Determine the next timestep
        h = h * 0.9 * (atol / err)**(1/3)

        #Check if this timestep would overstep our target time
        tau = loop_state["tau"]
        h = jax.lax.cond( tau + h > t, lambda _: t - tau, lambda _: h, None )

        accepted = loop_state["accepted"]
        rejected = loop_state["rejected"]

        #Decide if we accept the step or not
        #If we accept, we must update both x and tau
        xf, tau, accepted, rejected = jax.lax.cond( err < atol, lambda _: (xf, tau + h, accepted + 1, rejected), lambda _: (x, tau, accepted, rejected + 1), None )

        #jax.debug.print("tau = {}, h = {}, err = {}, atol = {}", tau, h, err, atol)

        return {"x": xf, "tau": tau, "h": h, "k1": k4, "fevals": fevals, "accepted": accepted, "rejected": rejected}

    loop_state = jax.lax.while_loop(cond_fun, body_fun, loop_state)
    return loop_state["x"], loop_state["fevals"], loop_state["accepted"], loop_state["rejected"]





if __name__ == "__main__":

    import time

    @jax.jit
    def vel( x ):
        sigma = 10
        rho = 28
        beta =8/3

        v = jnp.array( [sigma*(x[1]-x[0]), x[0]*(rho-x[2]) - x[1], x[0]*x[1] - beta*x[2]] )
        return v
    
    x = jnp.array([1.0, 2.0, 3.0])

    t = 2.0

    print("Testing adaptive integration on Lorenz attractor...\n")
    print("\nRunning without JIT")

    start = time.time()
    xf, fevals, acc, rej = runge_kutta_32(x, vel, t, h=1e-2, atol=1e-5)
    stop = time.time()

    print( f"xf = {xf} with fevals = {fevals}, walltime = {stop-start:.3f}, accepted = {acc}, rejected = {rej}" )





    print("\nRunning with JIT")

    rk32 = jax.jit(runge_kutta_32, static_argnames="vel_fn")
    _ = rk32(x, vel, t, h=1e-2, atol=1e-5)

    start = time.time()
    xf, fevals, acc, rej = rk32(x, vel, t, h=1e-2, atol=1e-5)
    stop = time.time()

    print( f"xf = {xf} with fevals = {fevals}, walltime = {stop-start:.3f}, accepted = {acc}, rejected = {rej}" )




    print("\nAttempting Jacobian-vector product")


    def wrapper( input_dict ):
        return rk32(input_dict["x"], vel, input_dict["t"], h=1e-2, atol=1e-5)[0]

    input_dict = {"x": x, "t": t}

    jacobian = jax.jit( lambda v: jax.jvp( wrapper, primals=(input_dict,), tangents=(v,) )[1] )

    _ = jacobian(input_dict)

    start = time.time()
    v = jnp.array([0.0, -3.0, 4.0])
    v_dict = {"x": v, "t": 4.0}    
    Jv = jacobian(v_dict)
    stop = time.time()
    
    print( f"Jv = {Jv}, walltime = {stop-start:.3f}" )




    print("\nAttempting vector-Jacobian product")

    _, jac_t = jax.vjp( wrapper, input_dict )

    jac_t = jax.jit(jac_t)

    w = jnp.array([1.0, 2.0, 3.0])
    _ = jac_t(w)

    start = time.time()
    wJ = jac_t(w)
    stop = time.time()
    
    print( f"wJ = {wJ}, walltime = {stop-start:.3f}" )