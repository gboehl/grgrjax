# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp

from .__version__ import __version__
from .helpers import *
from .newton import *

__all__ = ["newton_jax", "newton_jax_jit", "amax", "callback_func", "jax_print",
           "jvp_vmap", "vjp_vmap", "val_and_jacfwd", "val_and_jacrev"]


def jax_print(w):
    """Print in jax compiled functions. Wrapper around `jax.experimental.host_callback.id_print`.
    """
    return jax.experimental.host_callback.id_print(w)
