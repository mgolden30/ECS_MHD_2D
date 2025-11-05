## MATLAB scripts

While all of the numerics are performed in Python with JAX, I enjoy 
visualization in MATLAB. This does create a small pain point, which is
reading Python data into MATLAB. Thankfully, recent releases of MATLAB can
call Python functions. I will be writing all MATLAB code assuming release 
R2025b. You can check the supported versions of Python [here](https://www.mathworks.com/support/requirements/python-compatibility.html).
For R2025b, Python 3.12 is the latest supported version. 

For easy setup, install the desired Python version with conda.
```shell
conda create -n for_matlab python=3.12
conda activate for_matlab
```
Start MATLAB from terminal after activating this environment. You can check
that the environment is recognized by MATLAB by running 
```MATLAB
>> pyenv
``` 

