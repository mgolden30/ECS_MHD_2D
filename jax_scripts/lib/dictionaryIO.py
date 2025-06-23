'''
Define easy to use and general purpose functions for saving dictionary data. 
In particular, save parameter dictionaries with data dictionaries so the entire state is stored in one place.
'''

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

    input_dict = {k: jnp.array(loaded[k]) for k in input_keys}
    param_dict = {k: jnp.array(loaded[k]) for k in param_keys}

    return input_dict, param_dict