# ECS_MHD_2D

Python code to find Exact Coherent Structures (ECS) in 2D Magnetohydrodynamics (MHD)

# Code Organization

deprecated - a folder for files that are not (to my knowledge) actively used. These might be useful in the future.

derive_adjoint_looping - a folder with MATLAB code that derives an explicit integration scheme to carry out adjoint looping. While I am focusing on a JAX autodiff implementation, autodiff does not know about our memory limitations. It might be better to explicitly integrate the adjoint looping equations in a memory-conservative way. 

jax_scripts - python code for solving the MHD equations in JAX and doing optimization.

matlab_scripts - visualization scripts since I hate matplotlib with a passion

pytorch_scripts - I experimented with pytorch as a GPU accelerated numpy replacement. I used autodiff, although not that successfully. The pytorch autodiff is rigid and uncomfortable to use. JAX is much more mathematical in nature and easier to program.


