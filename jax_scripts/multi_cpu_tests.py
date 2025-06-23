'''
As the dimension of my Krylov subspace grows for Newton-Raphson, I feel the need to investigate the possibility of massively parallel Jacobian evaluations.
This script will help me investigate the scalability of JAX on many CPUs.
'''

import os


flags = os.environ.get("XLA_FLAGS", "")
flags += " --xla_force_host_platform_device_count=8" #simulate 8 devices
#enforce CPU-only execution
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["XLA_FLAGS"] = flags

#Force JAX to use CPUs
os.environ["JAX_PLATFORM_NAME"] = "cpu"

#import functools
#from typing import Any, Dict, Tuple
#import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
#from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


print( jax.devices() )

a = jnp.arange(8)
print("Array", a)
print("Device", a.device )
print("Sharding", a.sharding)

mesh = Mesh(np.array(jax.devices()), ("i",))
print(mesh)

sharding = NamedSharding( mesh, P("i"), )

a_sharded = jax.device_put(a, sharding)
print("Sharded array", a_sharded)
print("Device", a_sharded.devices() )
print("Sharding", a_sharded.sharding )

jax.debug.visualize_array_sharding(a_sharded)

out = jnp.tanh(a_sharded)
print(out)
jax.debug.visualize_array_sharding(out)
