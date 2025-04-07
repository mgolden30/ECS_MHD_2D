import numpy as np
import time

import mhd
import mhd_torch

n = 128 #gridpoints per side

#Make coordinate vectors
x = np.arange(0,n)/n*2*np.pi
x = np.reshape( x, [n,1] )
y = np.reshape( x, [1,n] )

#Declare initial data and forcing
fields = np.zeros([2,n,n])

fields[0,:,:] = np.sin(3*x+3*y)*15 #vorticity
fields[1,:,:] = 0*np.sin(y)*np.cos(x) #current

#Or, define random initial data
fields = mhd.random_initial_data( n, seed=0 )

#fields = np.load("late.npy")

#Forcing on vorticity
force = -4*np.cos(4*y)
mean_B = np.array( [0, 0.1] )
nu = 1/100
eta= 1/100

dt  = 0.0025
steps = round( 200 / dt)

deriv_matrices = mhd.fourier(n) #precompute derivative matrices in Fourier space
params = { "mean_B": mean_B, "nu": nu, "eta": eta, "force": force, "deriv_matrices": deriv_matrices }



#Integrate forward in time with Exponential Ansatz Runge-Kutta
start_time = time.time()
fields = mhd_torch.eark4( fields, dt, steps, params )
end_time = time.time()

print(f"Integration took {end_time - start_time} seconds")

#Save final vorticity and current
mhd.visualize( fields, "fields.png" )
np.save("late.npy", fields)