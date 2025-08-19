<div align="center">

# ECS_MHD_2D

### Exact Coherent Structures in 2D Magnetohydrodynamics

*A Python implementation for finding Exact Coherent Structures (ECS) in 2D MHD using JAX*

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-GPU%20Accelerated-orange.svg)](https://jax.readthedocs.io/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

<p align="center">
  <img src="figures/example.gif" alt="Demo animation" width="500"/>
</p>

<!--
## Table of Contents
- [Quick Start](#-quick-start)
- [Environment Setup](#️-environment-setup)
- [Tutorial](#-tutorial)
- [MATLAB Visualization](#-matlab-visualization)
- [Troubleshooting](#-troubleshooting)
- [Additional Resources](#-additional-resources)
-->
---

## Environment Setup

### Option 1: Use Provided Environment (Recommended)

```bash
conda env create -f environment.yml
```

### Option 2: Manual Installation

```bash
# Create and activate conda environment
conda create -n ECS_MHD python=3.13
conda activate ECS_MHD
pip install -U "jax[cuda12]"
pip install matplotlib
pip install imageio[ffmpeg]
```

> ** Note:** This assumes CUDA drivers version 12.x. Consult the [JAX documentation](https://jax.readthedocs.io/) for other versions.

**GPU vs CPU:** While the code runs significantly faster on GPU, it can also run on CPU-only systems.

---

##  Tutorial

### Step 1: Generate Turbulent Dataset

Start by creating turbulent data:

```bash
python jax_scripts/generate_turbulence.py
```

**Expected output:**
```
Transient of 8192 steps took 7.515015125274658 second...
Generating 256 steps of turbulence took 14.76011872291565 seconds.
Recurrence diagram computed in 2.2505459785461426 seconds
```
This creates two files. 
temp_data/turb.npz stores all grid parameters and a bit of turbulence to intialize RPO guesses with.
temp_data/dist.mat contains a recurrence diagram. This can be visualized in MATLAB or loaded into any python script with scipy.io.loadmat.

### Step 2: Create Initial RPO Guess

Make an initial guess at a Relative Periodic Orbit:

1. Specify the array `idx=[initial,final]` (MATLAB indexing!) in `jax_scripts/adjoint_descent.py`
2. Run the adjoint descent:

```bash
python jax_scripts/adjoint_descent.py
```

**Expected output:**
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
⋮
```
This script will effectively run until user-terminated. The state is only saved every 64 steps in temp_data/adjoint_descent/.


### Step 3: Fine-tune with Newton-GMRES

Once the error is sufficiently small, refine the solution with Newton-Raphson iteration:

1. Edit `jax_scripts/newton.py` to specify the input file
2. Run the script

```bash
python jax_scripts/newton.py
```

**Expected output:**
```
[CudaDevice(id=0)]
Choosing single shooting with adaptive timestepping:
Evaluating objective: 0.223 seconds
Evaluating Jacobian: 0.415 seconds
Transpose walltime = 1.235
Iteration 0: rel_err=3.308e-01, |f|=2.195e+02, fwall=0.222, gmreswall=30.739, gmres_rel_res=2.011e-02, damp=1.000e+00, T=4.602e+00, sx=1.218e-01
Iteration 1: rel_err=1.708e-01, |f|=1.134e+02, fwall=0.232, gmreswall=29.471, gmres_rel_res=1.022e-03, damp=1.000e+00, T=4.601e+00, sx=1.379e-01
Iteration 2: rel_err=1.223e-01, |f|=8.101e+01, fwall=0.235, gmreswall=28.989, gmres_rel_res=1.211e-02, damp=1.000e+00, T=4.628e+00, sx=1.504e-01
⋮
```
This script will effectively run until user-terminated. The states will be saved in temp_data/newton/ in the usual format.


### Step 4: Visualize Solutions

If you want to visualize a particular solution, use jax_scripts/animate.py after specifying the input file.
```bash
python jax_scripts/animate.py
```
This creates the video figures/RPO.mp4.

Currently Newton hunts for RPOs with adaptive timestepping and animate.py uses fixed timestepping. This might matter for some violent solutions.

**Happy hunting!**


<div align="center">


</div>
