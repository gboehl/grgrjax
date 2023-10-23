# -*- coding: utf-8 -*-

import jax
import time
import jax.numpy as jnp
from jax._src.api import (_check_input_dtype_jacfwd, _check_input_dtype_jacrev, _check_output_dtype_jacfwd, _check_output_dtype_jacrev, _ensure_index, _jvp,
                          _vjp, _std_basis, _jacfwd_unravel, _jacrev_unravel, lu, argnums_partial, tree_map, tree_structure, tree_transpose, partial, Callable, Sequence, vmap)
# fix import location for jax 0.4.1
try:
    from jax._src.api import _check_callable
except ImportError:
    from jax._src.api_util import check_callable as _check_callable


def amax(x, return_arg=False):
    """Return the maximum absolute value.
    """
    absx = jnp.abs(x)
    if return_arg:
        arg = jnp.argmax(absx)
        return absx[arg], arg
    else:
        return absx.max()


def jvp_vmap(fun: Callable, argnums=0):
    """Vectorized (forward-mode) jacobian-vector product of ``fun``. This is by large adopted from the implementation of jacfwd in jax._src.api.

    Args:
      fun: Function whose value and Jacobian is to be computed.
      argnums: Optional, integer or sequence of integers. Specifies which positional argument(s) to differentiate with respect to (default ``0``).

    Returns:
      A function with the same arguments as ``fun``, that evaluates the value and Jacobian of ``fun`` using forward-mode automatic differentiation.
    """
    _check_callable(fun)
    argnums = _ensure_index(argnums)

    def jvpfun(args, tangents, **kwargs):

        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False)
        pushfwd = partial(_jvp, f_partial, dyn_args)
        y, jac = vmap(pushfwd, out_axes=(None, -1), in_axes=-1)(tangents)

        return y, jac

    return jvpfun


def vjp_vmap(fun: Callable, argnums=0):
    """Vectorized (reverse-mode) vector-jacobian product of ``fun``. This is by large adopted from the implementation of jacrev in jax._src.api.

    Args:
      fun: Function whose value and Jacobian are to be computed.
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default ``0``).

    Returns:
      A function with the same arguments as ``fun``, that evaluates the value and Jacobian of
      ``fun`` using reverse-mode automatic differentiation.
    """
    _check_callable(fun)

    def vjpfun(args, tangents, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        y, pullback = _vjp(f_partial, *dyn_args)
        jac = vmap(pullback)(tangents)
        return y, jac

    return vjpfun


def val_and_jacfwd(fun: Callable, argnums=0, has_aux: bool = False, holomorphic: bool = False) -> Callable:
    """Value and Jacobian of ``fun`` evaluated column-by-column using forward-mode AD. Apart from returning the function value, this is one-to-one adopted from jax._src.api.

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
      A function with the same arguments as ``fun``, that evaluates the value and Jacobian of
      ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
      then a pair of (jacobian, auxiliary_data) is returned.
    """
    _check_callable(fun)
    argnums = _ensure_index(argnums)

    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
        if not has_aux:
            pushfwd = partial(_jvp, f_partial, dyn_args)
            y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
        else:
            pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = vmap(pushfwd, out_axes=(
                None, -1, None))(_std_basis(dyn_args))
        tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        if not has_aux:
            return y, jac_tree
        else:
            return y, jac_tree, aux

    return jacfun


def val_and_jacrev(fun: Callable, argnums=0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> Callable:
    """Value and Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD. Apart from returning the function value, this is one-to-one adopted from jax._src.api.

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
      A function with the same arguments as ``fun``, that evaluates the value and Jacobian of
      ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
      then a pair of (jacobian, auxiliary_data) is returned.
    """
    _check_callable(fun)

    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacrev,
                 holomorphic, allow_int), dyn_args)
        if not has_aux:
            y, pullback = _vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
        jac = vmap(pullback)(_std_basis(y))
        jac = jac[0] if isinstance(argnums, int) else jac
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
        jac_tree = tree_transpose(tree_structure(
            example_args), tree_structure(y), jac_tree)
        if not has_aux:
            return y, jac_tree
        else:
            return y, jac_tree, aux

    return jacfun
