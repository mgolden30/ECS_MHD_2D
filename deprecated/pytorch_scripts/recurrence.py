import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat

import mhd
import mhd_torch



n = 128 #gridpoints per side

#Make coordinate vectors
x = np.arange(0,n)/n*2*np.pi
x = np.reshape( x, [n,1] )
y = np.reshape( x, [1,n] )

#Forcing on vorticity
force = -4*np.cos(4*y)
mean_B = np.array( [0, 0.1] )
nu = 1/100
eta= 1/100

#Load something we;ve integrated forwrd in time quite a bit
fields = np.load("late.npy")

deriv_matrices = mhd.fourier(n) #precompute derivative matrices in Fourier space
params = { "mean_B": mean_B, "nu": nu, "eta": eta, "force": force, "deriv_matrices": deriv_matrices, "n": n }



##############################
# Part 1: recurrence diagram
##############################

dt = 0.002
every = 100
m = 256 #number of snapshots

my_dict = loadmat("traj.mat" )



idx = [56, 69]

#guess a period
steps = every*(idx[1] - idx[0])
print(steps)
dt = 2*dt
steps = steps//2
T = steps*dt
theta = 0
fields = my_dict["traj"][:,:,:,idx[0]-1]



fields, T, theta = mhd_torch.adjoint_descent(fields, T, theta, steps, params, maxit=128, lr=0.1)

print(T)
print(theta)

mhd.visualize(fields, "ECS.png")

#save output
np.save("f", fields ) 
np.save("T", T )
np.save("theta", theta )