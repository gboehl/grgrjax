# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp

from .helpers import *
from .newton import *
from jax.experimental.host_callback import id_print as jax_print
amax = jax.jit(lambda x: jnp.abs(x).max())

__version__ = '0.1.1'
