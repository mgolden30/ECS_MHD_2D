'''
Let's check that our numerical scheme is actually fourth order
'''

import numpy as np
import mhd
import mhd_torch
import torch

#gridpoints per side
n = 128

#Make coordinate vectors
x = np.arange(0,n)/n*2*np.pi
x = np.reshape( x, [n,1] )
y = np.reshape( x, [1,n] )

#Define random initial data
fields = mhd.random_initial_data( n, seed=0 )

#Forcing on vorticity
force = -4*np.cos(4*y)
mean_B = np.array( [0, 0.1] )
nu = 1/100
eta= 1/100

T = 1.0
steps = np.array( [128, 256, 512] )
steps_fine = steps[-1]*2

deriv_matrices = mhd.fourier(n) #precompute derivative matrices in Fourier space
params = { "mean_B": mean_B, "nu": nu, "eta": eta, "force": force, "deriv_matrices": deriv_matrices }


integrator = mhd_torch.eark4
#integrator = mhd.eark4

#Integrate the fine timestep case
fields_fine = integrator( fields, T/steps_fine, steps_fine, params )
err = 0.0*steps

#Integrate forward in time with Exponential Ansatz Runge-Kutta
for i in range(len(steps)):
    fields_out = integrator( fields, T/steps[i], steps[i], params )
    err[i] = np.linalg.norm( np.reshape( fields_out - fields_fine, -1) )  
    mhd.visualize( fields_out, filename=f"{steps[i]}.png" )


#Fit to a power law
coeffs = np.polyfit( np.log(steps), np.log(err), 1)
print(err)
print(steps)
print(coeffs)