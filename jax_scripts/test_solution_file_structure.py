import jax
import jax.numpy as jnp
import lib.dictionaryIO as dictionaryIO

import sys

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re40/RPO1.npz")
input_dict, param_dict = dictionaryIO.load_dicts("temp_data/newton/1.npz")


def inspect_dict(d):
    for key, value in d.items():
        if hasattr(value, 'size') and hasattr(value, 'dtype'):
            size_bytes = value.size * value.dtype.itemsize
            size_kb = size_bytes / (1024**1)
            print(f"Key: {key}, shape: {value.shape}, dtype: {value.dtype}, size: {size_kb:.2f} KB")
        else:
            print(f"Key: {key}, type: {type(value)}, size: unknown")

print("input_dict:")
inspect_dict(input_dict)

print("\nparam_dict:")
inspect_dict(param_dict)
