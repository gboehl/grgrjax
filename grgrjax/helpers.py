# -*- coding: utf-8 -*-

import jax
import time
import jax.numpy as jnp
from jax._src.api import (_check_input_dtype_jacfwd, _check_input_dtype_jacrev, _check_output_dtype_jacfwd, _check_output_dtype_jacrev, _ensure_index, _jvp,
                          _vjp, _std_basis, _jacfwd_unravel, _jacrev_unravel, lu, argnums_partial, tree_map, tree_structure, tree_transpose, partial, Callable, Sequence, vmap, debug_info)
from jax._src.api_util import check_callable


def amax(x, return_arg=False):
    """Return the maximum absolute value.
    """
    absx = jnp.abs(x)
    if return_arg:
        arg = jnp.argmax(absx)
        return absx[arg], arg
    else:
        return absx.max()


def jvp_vmap(fun: Callable, argnums=0, has_aux: bool = False, holomorphic: bool = False) -> Callable:
    """Vectorized (forward-mode) Jacobian-vector product of ``fun``. This is by large adopted from the implementation of jacfwd in jax._src.api.

    Args:
      fun: Function whose value and Jacobian is to be computed.
      argnums: Optional, integer or sequence of integers. Specifies which positional argument(s) to differentiate with respect to (default ``0``).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
      holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
        holomorphic. Default False.
      allow_int: Optional, bool. Whether to allow differentiating with
        respect to integer valued inputs. The gradient of an integer input will
        have a trivial vector-space dtype (float0). Default False.

    Returns:
      A function with the same arguments as ``fun``, that evaluates the vectorized Jacobian-vector product of 
      ``fun`` using forward-mode automatic differentiation. If ``has_aux`` is True
      then auxiliary_data is returned as last argument.
    """
    check_callable(fun)
    argnums = _ensure_index(argnums)

    def jvpfun(args, tangents, **kwargs):

        try:
            f = lu.wrap_init(fun, kwargs)
        except TypeError:
            f = lu.wrap_init(fun, kwargs, debug_info=debug_info("jvp_vmap", fun, args, kwargs, static_argnums=(argnums,) if isinstance(argnums, int) else argnums))
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        if tangents is None: 
            tangents = _std_basis(dyn_args)
        tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
        if not has_aux:
            pushfwd = partial(_jvp, f_partial, dyn_args)
            y, jac = vmap(pushfwd, out_axes=(None, -1))(tangents)
        else:
            pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = vmap(pushfwd, out_axes=(
                None, -1, None))(tangents)
        if tangents is None: 
            tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
            example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
            jac = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        if not has_aux:
            return y, jac
        else:
            return y, jac, aux

    return jvpfun


def val_and_jacfwd(*args, **kwargs):
    """Wrapper around jvp_vmap to evaluate Value and Jacobian of ``fun`` column-by-column using forward-mode AD. 
    """
    jvpfun = jvp_vmap(*args, **kwargs)

    def jacfun(*args, **kwargs):
        return jvpfun(args=args, tangents=None, **kwargs)

    return jacfun


def vjp_vmap(fun: Callable, argnums=0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> Callable:
    """Vectorized (reverse-mode) vector-Jacobian product of ``fun``. This is by large adopted from the implementation of jacrev in jax._src.api.

    Args:
      fun: Function whose value and Jacobian are to be computed.
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default ``0``).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
      holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
        holomorphic. Default False.
      allow_int: Optional, bool. Whether to allow differentiating with
        respect to integer valued inputs. The gradient of an integer input will
        have a trivial vector-space dtype (float0). Default False.

    Returns:
      A function with the same arguments as ``fun``, that evaluates the vectorized vector-Jacobian product of 
      ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
      then auxiliary_data is returned as last argument.
    """
    check_callable(fun)

    def vjpfun(args, tangents, **kwargs):
        try:
            f = lu.wrap_init(fun, kwargs)
        except TypeError:
            f = lu.wrap_init(fun, kwargs, debug_info=debug_info("vjp_vmap", fun, args, kwargs, static_argnums=(argnums,) if isinstance(argnums, int) else argnums))
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacrev,
                 holomorphic, allow_int), dyn_args)
        if not has_aux:
            y, pullback = _vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
        if tangents is not None:
            jac = vmap(pullback)(tangents)
        else:
            jac = vmap(pullback)(_std_basis(y))
            jac = jac[0] if isinstance(argnums, int) else jac
            example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
            jac = tree_map(partial(_jacrev_unravel, y), example_args, jac)
            jac = tree_transpose(tree_structure(
                example_args), tree_structure(y), jac)
        if not has_aux:
            return y, jac
        else:
            return y, jac, aux

    return vjpfun


def val_and_jacrev(*args, **kwargs):
    """Wrapper around vjp_vmap to evaluate Value and Jacobian of ``fun`` row-by-row using reverse-mode AD. 
    """

    vjpfun = vjp_vmap(*args, **kwargs)

    def jacfun(*args, **kwargs):
        return vjpfun(args=args, tangents=None, **kwargs)

    return jacfun
