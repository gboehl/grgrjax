# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp

from .helpers import *
from .newton import *

__version__ = '0.4.0'
__all__ = ["newton_jax", "newton_jax_jit", "amax", "callback_func", "jax_print",
           "jvp_vmap", "vjp_vmap", "val_and_jacfwd", "val_and_jacrev"]


def jax_print(w):
    """Print in jax compiled functions. Wrapper around `jax.experimental.host_callback.id_print`.
    """
    return jax.experimental.host_callback.id_print(w)


def amax(x):
    """Return the maximum absolute value.
    """
    return jnp.abs(x).max()
