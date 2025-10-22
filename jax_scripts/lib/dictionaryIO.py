'''
Define easy to use and general purpose functions for saving dictionary data. 
In particular, save parameter dictionaries with data dictionaries so the entire state is stored in one place.
'''

import lib.mhd_jax as mhd_jax

import jax.numpy as jnp
import numpy as np

def save_dicts( filename, input_dict, param_dict ):
    #Combine the two dictionaries into a single dictionary
    combined_dict = input_dict.copy()
    combined_dict.update(param_dict)

    # Add keys to remember which data came from which dict
    input_keys = list(input_dict.keys())
    param_keys = list(param_dict.keys())

    combined_dict['input_keys'] = input_keys
    combined_dict['param_keys'] = param_keys

    #Convert to default numpy from jnp
    data_np = {k: np.array(v) for k, v in combined_dict.items()}

    #Save. ** does dictionary unpacking
    np.savez( filename, **data_np )


def load_dicts(filename):
    loaded = np.load(filename, allow_pickle=True)

    input_keys = loaded['input_keys']
    param_keys = loaded['param_keys']

    def safe_convert(val):
        # If it's an ndarray with a numeric dtype, convert to jnp.array
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
            return jnp.array(val)
        # Otherwise leave it as-is (string or object)
        else:
            return val.item() if val.shape == () else val


    input_dict = {k: safe_convert(loaded[k]) for k in input_keys}
    param_dict = {k: safe_convert(loaded[k]) for k in param_keys}

    return input_dict, param_dict



def remove_grid_information(param_dict):
    '''
    I typically save both input_dict and param_dict for a solution. 
    The problem is that param_dict contains grid tensors that are memory bloat
    '''
    pd = param_dict.copy()
    for key in ['x','y','kx','ky','mask','to_u','to_v','inv_lap','forcing']:
        pd.pop(key, None)  # remove if exists
    return pd



def recompute_grid_information(input_dict, param_dict):
    '''
    The inverse of remove_grid_information
    '''
    #Assume grid is square
    n = input_dict['fields'].shape[-1]
    data_type = input_dict['fields'].dtype
    print(data_type)
    param_dict.update( mhd_jax.construct_domain(n, data_type) )
    
    forcing_fn = eval(param_dict['forcing_str'])
    x = param_dict['x']
    y = param_dict['y']
    param_dict['forcing'] = forcing_fn(x,y)
    return param_dict