# ECS_MHD_2D

Python code to find Exact Coherent Structures (ECS) in 2D Magnetohydrodynamics (MHD)

# Setting up the environment

All convergence is critically dependent on JAX. To install, do one of the following.

1. Try to use my snapshot environment.yml provided here.
$ conda env create -f environment.yml

2. Replicate my simple installation process.
$ conda create -n ECS_MHD python=3.13
$ conda activate ECS_MHD
$ pip install -U "jax[cuda12]"
$ pip install matplotlib

This assumes your CUDA drivers are version 12.x. Consult the JAX documentation if this does not apply to you.
Of course, the code runs significantly faster on a GPU, but a GPU is not explicitly required. You can run these scripts on CPU only.

# Tutorial

All python code is expected to be ran from this directory. For example,

python jax_scripts/newton.py


Roughly speaking, hunting for Relative Periodic Orbits can be done with the following steps.
1. Generate a turbulent dataset using
```
$ python jax_scripts/generate_turbulence.py
```

on my machine, I get the following output:
```
Transient of 8192 steps took 7.515015125274658 second...
Generating 256 steps of turbulence took 14.76011872291565 seconds.
Recurrence diagram computed in 2.2505459785461426 seconds
```
You can edit this script to change parameters of the flow and numerical grid. 

2. Make an initial guess at an RPO (even a bad guess is okay). You do this by specifying the array idx=[initial,final] (MATLAB indexing!) in jax_scripts/adjoint_descent.py.
Once you have an initial guess, run
```
$ python jax_scripts/adjoint_descent.py
```
My machine outputs the following:
```
Creating new RPO guess from temp_data/turb.npz...
using 1088 timesteps of type <class 'int'> 
0: loss=2.239609, walltime=3.824, T=4.260, sx=0.010, completed=True, fevals=1085, accepted=201, rejected=70
1: loss=2.062645, walltime=1.739, T=4.270, sx=0.020, completed=True, fevals=1181, accepted=206, rejected=89
2: loss=1.903478, walltime=1.714, T=4.280, sx=0.030, completed=True, fevals=1153, accepted=204, rejected=84
3: loss=1.759379, walltime=1.694, T=4.290, sx=0.040, completed=True, fevals=1125, accepted=202, rejected=79
4: loss=1.628692, walltime=1.676, T=4.300, sx=0.050, completed=True, fevals=1097, accepted=200, rejected=74
5: loss=1.510585, walltime=1.655, T=4.310, sx=0.059, completed=True, fevals=1069, accepted=198, rejected=69
6: loss=1.405015, walltime=1.483, T=4.320, sx=0.068, completed=True, fevals=833, accepted=194, rejected=14
```

3. Once the error is sufficiently small (whatever that means to you), we can fine tune the state with Newton-GMRES. Edit jax_scripts/newton.py to tell it which file to load in and run
```
$ python jax_scripts/newton.py
```

My machine outputs the following:
```
[CudaDevice(id=0)]
Choosing single shooting with adaptive timestepping:
Evaluating objective: 0.223 seconds
Evaluating Jacobian: 0.415 seconds
Transpose walltime = 1.235
Iteration 0: rel_err=3.308e-01, |f|=2.195e+02, fwall=0.222, gmreswall=30.739, gmres_rel_res=2.011e-02, damp=1.000e+00, T=4.602e+00, sx=1.218e-01
Iteration 1: rel_err=1.708e-01, |f|=1.134e+02, fwall=0.232, gmreswall=29.471, gmres_rel_res=1.022e-03, damp=1.000e+00, T=4.601e+00, sx=1.379e-01
Iteration 2: rel_err=1.223e-01, |f|=8.101e+01, fwall=0.235, gmreswall=28.989, gmres_rel_res=1.211e-02, damp=1.000e+00, T=4.628e+00, sx=1.504e-01
```

Happy hunting!

# MATLAB Visualization

I prefer MATLAB for data visualization, so matlab_scripts is full of various code for visualizing the state. It is not required by any means, but it is how I use this code.
See matlab_scripts/recurrence.m to look at a nice recurrence diagram and animate turbulence.
