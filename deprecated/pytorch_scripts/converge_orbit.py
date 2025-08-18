import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter
from scipy.io import savemat

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
params = { "mean_B": mean_B, "nu": nu, "eta": eta, "force": force, "deriv_matrices": deriv_matrices }



##############################
# Part 1: recurrence diagram
##############################

dt = 0.002
every = 100
m = 256#number of snapshots

#store the trajectory
traj = np.zeros([2,n,n,m])
traj[:,:,:,0] = fields

for i in range(m-1):
    print(i)
    traj[:,:,:,i+1] = mhd_torch.eark4( traj[:,:,:,i], dt, every, params )
    #mhd.visualize(traj[:,:,:,i], f"frames/{i}.png")

traj2 = np.fft.fft2(traj, axes=[1,2])

dist = np.zeros([m,m])
traj2 = traj2.reshape( [-1, m] )


scale = np.linalg.norm( np.abs(traj2[:,0]))
for i in range(m):
    for j in range(m):
        dist[i,j] = np.linalg.norm( np.abs(traj2[:,i]) - np.abs(traj2[:,j]) ) / scale

plt.imshow(dist, vmax=0.5)
plt.colorbar()
plt.savefig("dist.png")


my_dict = {"dist": dist, "traj": traj}
savemat("traj.mat", my_dict )

exit()

#guess a period
T = 3.0

#fields = np.load("f.npy")
#T = np.load("T.npy")

steps = 512
fields, T = mhd_torch.adjoint_descent(fields, T, steps, params, maxit=128, lr=1e-2)

mhd.visualize(fields, "ECS.png")

#save output
np.save("f", fields ) 
np.save("T", T )