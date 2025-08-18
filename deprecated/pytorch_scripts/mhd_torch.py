'''
Give pytorch implementations of evolution functions. Keep initial data + derivative matrices in base MHD library.

Version 1.0

'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.adagrad

#Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}...")

#Define the pytorch functions for ffts
rfft2 = torch.fft.rfft2
irfft2= torch.fft.irfft2
add_dim= torch.unsqueeze

def state_velocity( fields, force, deriv_matrices, mean_B, fourier_input=True, fourier_output=True ):
    '''
    PURPOSE:
    Compute the non-dissipative terms in the dynamics

    INPUT:
    fields - a tensor of shape [2,n,n]

    '''

    if not fourier_input:
        #We must transform to Fourier space
        fields = rfft2(fields)

    #uncurl the fields to get flow velocity and magnetic field
    vectors = 1j * deriv_matrices["uncurl"] * add_dim(fields, 1)
    vectors = irfft2(vectors)

    #Account for mean magnetic field
    vectors[1,0,:,:] = vectors[1,0,:,:] + mean_B[0]
    vectors[1,1,:,:] = vectors[1,1,:,:] + mean_B[1]

    #Compute the gradients
    grads = 1j * deriv_matrices["ki"] * add_dim(fields, 1)
    grads = irfft2(grads)

    #Lets compute u dot grad(omega) - B dot grad(j) as "advection"
    advection = torch.sum(vectors * grads, axis=1)

    #advection, magentic stress, and forcing
    dwdt = - advection[0,:,:] + advection[1,:,:] + force

    #dealias the time derivative!
    dwdt = deriv_matrices["mask"] * rfft2( dwdt)

    #u times B
    djdt = vectors[0,0,:,:]*vectors[1,1,:,:] - vectors[0,1,:,:]*vectors[1,0,:,:]
    #Take Laplacian
    djdt = deriv_matrices["k_sq"] * deriv_matrices["mask"] * rfft2(djdt)
    
    #combine these into a single tensor
    time_deriv = torch.stack( (dwdt, djdt) )
    time_deriv = time_deriv.reshape( fields.shape )

    if not fourier_output:
        #We must transform back to real space
        time_deriv = irfft2(time_deriv)

    return time_deriv



def eark4( fields, dt, steps, params, numpy_output=True ):
    '''
    Exponential Ansatz Runge-Kutta 4th order
    '''

    #If we passed a numpy array, change to torch tensor and move to GPU
    if isinstance(fields, np.ndarray):
        fields = torch.from_numpy(fields).to(device)

    dict    = params["deriv_matrices"]
    mean_B = params["mean_B"]
    force  = params["force"]

    #Move derivative matrices onto GPU if needed
    for key, item in dict.items():
        if isinstance(item, np.ndarray):
            # Historical note: this line took me a long time to correctly write.
            # It is very important to cast all of the derivative matrices (especially integer valued ones)
            # as double precision for some reason
            dict[key] = torch.from_numpy(item.astype(np.float64)).to(device)
            print( f"{key}: {dict[key].dtype}" )
    if isinstance(force, np.ndarray):
        force = torch.from_numpy(force).to(device)

    #dissipation exponential (clean this up eventually)
    L = dict["k_sq"] * torch.tensor( [params["nu"], params["eta"]] ).reshape([2,1,1]).to(device)
    e = torch.exp( -dt/2 * L )

    #Assume input is real. Map to Fourier space 
    fields = rfft2(fields)

    for _ in range(steps):
        k1 = dt*state_velocity( fields, force, dict, mean_B )
        
        fields = fields * e #1
        k1     = k1     * e #1 
        k2 = dt*state_velocity( fields + k1/2, force, dict, mean_B )
        k3 = dt*state_velocity( fields + k2/2, force, dict, mean_B )
        
        fields = fields * e #2
        k1     = k1     * e #2 
        k2     = k2     * e #1 
        k3     = k3     * e #1 
        k4 = dt*state_velocity( fields + k3, force, dict, mean_B )

        fields = fields + (k1 + 2*(k2 + k3) + k4)/6
    
    #Back to real space
    fields = irfft2(fields)

    if numpy_output:
        #Return to cpu
        fields = fields.cpu().numpy()

    return fields



def adjoint_descent(fields, T, theta, steps, params, maxit=128, lr=1e-3):
    '''
    Converge the cost function (u(T) - u(0))*(T+1)/T, which blows up at T=0
    '''

    f = torch.tensor(fields, dtype = torch.float64).to(device)
    T = torch.tensor(T, dtype=torch.float64).to(device)
    theta = torch.tensor(theta, dtype=torch.float64).to(device)

    f.requires_grad = True
    T.requires_grad = True
    theta.requires_grad = True
    
    optimizer = torch.optim.AdamW([f, T, theta], lr=lr)

    n = params["n"]
    k = torch.arange(n)
    k[k>n] = k[k>n] - n
    k = torch.reshape(k, [1,n,1] ).to(device)

    for iter in range(maxit):
        #Evolve in time
        f_T = eark4( fields, T/steps, steps, params, numpy_output=False )
        
        #Shift in x
        f_T = torch.fft.fft(f_T, dim=1)
        f_T = torch.real(torch.fft.ifft( f_T * torch.exp(1j*k*theta) , dim=1))

        #Look at mismatch
        loss = (f_T - f)/T
        loss = torch.mean(torch.abs( loss ))

        print(f"step {iter}: loss = {torch.mean(torch.abs(loss))}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    f = f.cpu().detach().numpy()
    T = T.cpu().detach().numpy()
    theta = theta.cpu().detach().numpy()

    return f, T, theta