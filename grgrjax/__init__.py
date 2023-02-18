# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp

from .helpers import jvp_vmap, vjp_vmap, val_and_jacfwd, val_and_jacrev as val_and_jacrev
from .newton import callback_func, newton_jax, newton_jax_jit
from jax.experimental.host_callback import id_print as jax_print

__version__ = '0.1.1'


def amax(x):
    """return the maximum absolute value
    """
    return jnp.abs(x).max()
