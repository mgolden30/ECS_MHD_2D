'''
PURPOSE:
I want to write Newton-Raphson iteration for Rossler flow with jax
'''

import time
import jax
import jax.numpy as jnp

param_dict = {'a': 0.2, 'b': 0.2, 'c': 5.7}

def velocity( x, param_dict ):
    a = param_dict['a']
    b = param_dict['b']
    c = param_dict['c']

    vx = -x[1]-x[2]
    vy = x[0] + a*x[1]
    vz = b + x[2]*(x[0]-c)

    v = jnp.stack((vx,vy,vz))
    return v

x = jnp.stack( (1.0, 2.0, 3.0) )

print(x.shape)

v = velocity(x, param_dict)

print(v)
print(v.shape)


def rk4_step( x, dt, param_dict ):
    v = lambda x: velocity(x, param_dict) #Hide parameters in a lambda
    k1 = dt * v(x)
    k2 = dt * v(x + k1/2)
    k3 = dt * v(x + k2/2)
    k4 = dt * v(x + k3)

    x = x + (k1 + 2*k2 + 2*k3 + k4)/6
    return x

#Integrate a while to get onto the attractor
dt = 0.01
steps = 1024*8

jit_rk4_step = jax.jit(rk4_step)

for _ in range(steps):
    x = jit_rk4_step( x, dt, param_dict )

print(x)

#Guess a period
T = 6.0

#Append steps to the param_dict
steps = 256
param_dict.update({'steps': steps})
 
def periodic_orbit_objective( input_dict, param_dict ):
    #Pull out x and T
    x = input_dict['x']
    T = input_dict['T']
    steps = param_dict['steps']
    
    dt = T/steps

    #Compile the rk4 step since we will call it a lot
    jit_rk4_step = jax.jit(rk4_step)
    update_x = lambda i, x: jit_rk4_step(x, dt, param_dict)

    #Integrate with RK4
    xf = jax.lax.fori_loop( 0, steps, update_x, x)

    mismatch = xf - x

    #To make the system square and well-suited for power iteration, let's add a phase condition.
    #Why not \dot{x} = 0
    phase = -x[1] - x[2]

    #Make sure phase is an array rather than a value
    phase = jnp.reshape(phase, [1,])

    objective = jnp.concatenate( (mismatch, phase) )
    return objective

input_dict = {'x': x, 'T': T}
f = periodic_orbit_objective( input_dict, param_dict )
print(f)


#Try to compute the Jvp
as_input_dict = lambda f: {'x': f[0:3], 'T': f[3]}
objective = lambda input_dict: periodic_orbit_objective( input_dict, param_dict )

Jf = jax.jvp( objective, primals=(input_dict,), tangents=(as_input_dict(f),) )[1]
#print(f2)
print(Jf)

#Define a Jacobian lambda
jac = lambda v: jax.jvp( objective, primals=(input_dict,), tangents=(as_input_dict(v),) )[1]

inner = 3
ds = jax.scipy.sparse.linalg.gmres( jac, f, maxiter=inner )
print(ds)


# This is well and good, but I want to jit things for speed. 
# 
jit_jac = jax.jit(jac)

start = time.time()
for _ in range(8):
    _ = jac(f)
walltime = time.time() - start
print(f"non-jit jacobian took {walltime} seconds")


_ = jit_jac(f)
start = time.time()
for _ in range(8):
    _ = jit_jac(f)
walltime = time.time() - start
print(f"jit jacobian took {walltime} seconds")


jit_objective = jax.jit(objective)

damp = 0.1

def newton_step( input_dict ):
    f = jit_objective( input_dict )
    inner = 3
    ds, _ = jax.scipy.sparse.linalg.gmres( jit_jac, f, maxiter=inner )
    input_dict.update( {'x': input_dict['x'] - damp*ds[0:3], 'T': input_dict['T'] - damp*ds[3]  } )
    return input_dict, f

jit_newton_step = jax.jit( newton_step )

for it in range(128):
    input_dict, f = jit_newton_step( input_dict )
    print(f"iteration {it}: |f| = {jnp.linalg.norm(f)}, state = {input_dict}")