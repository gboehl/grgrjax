# -*- coding: utf-8 -*-

import jax
import time
import jax.numpy as jnp
import scipy.sparse as ssp
from .helpers import val_and_jacfwd, amax

try:
    ssp_csr_array = ssp.csr_array
    ssp_lil_array = ssp.lil_array
except AttributeError:
    ssp_csr_array = ssp._arrays.csr_array
    ssp_lil_array = ssp._arrays.lil_array


def _newton_cond_func(carry):
    (xi, eps, cnt), (func, verbose, maxit, tol) = carry
    cond = cnt < maxit
    cond = jnp.logical_and(cond, eps > tol)
    cond = jnp.logical_and(cond, ~jnp.isnan(eps))
    verbose = jnp.logical_and(cnt, verbose)
    jax.debug.callback(callback_func, cnt, eps, verbose=verbose)
    return cond


def _newton_body_func(carry):
    (xi, eps, cnt), (func, verbose, maxit, tol) = carry
    xi_old = xi
    f, jac = func(xi)
    xi -= jax.scipy.linalg.solve(jac, f)
    eps = amax(xi-xi_old)
    return (xi, eps, cnt+1), (func, verbose, maxit, tol)


def callback_func(cnt, err, *args, fev=None, ltime=None, verbose=True):
    """Print a formatted on-line update for a iterative process.
    """
    mess = f'    Iteration {cnt:2d}'
    if fev is not None:
        mess += f' | fev. {fev:3d}'
    mess += f' | error {err:.2e}'
    for misc in args:
        mess += misc
    if ltime is not None:
        mess += f' | lapsed {ltime:3.4f}s'
    if verbose:
        print(mess)


@jax.jit
def newton_jax_jit(func, init, maxit=30, tol=1e-8, verbose=True):
    """Newton method for root finding of `func` using automatic differentiation with jax and running in and as jitted jax.

    Parameters
    ----------
    func : callable
        Function returning (y, jac) where f(x)=y=0 should be found and jac is the jacobian. Must be jittable with jax. Could e.g. be the output of jacfwd_and_val. The function must be a jax.
    init : array
        Initial values of x
    maxit : int, optional
        Maximum number of iterations
    tol : float, optional
        Required tolerance. Defaults to 1e-8

    Returns
    -------
    xopt : array
        Solution value x for f
    (fopt, jacopt): tuple of arrays
        : value (`fopt`) and Jacobian (`jacopt`) of `func` at `xopt`
    niter : int
        Number of iterations
    success: bool
        Wether the convergence criterion was reached
    """
    (xi, eps, cnt), _ = jax.lax.while_loop(_newton_cond_func,
                                           _newton_body_func, ((init, 1., 0), (func, verbose, maxit, tol)))
    return xi, func(xi), cnt, eps > tol


def _perform_checks_newton(res, eps, cnt, jac_is_nan, tol, rtol, maxit):

    if jac_is_nan.any():
        res['success'] = False
        res['message'] = "The Jacobian contains NaNs."
        return True

    if eps < tol or eps < rtol:
        res['success'] = True
        res['message'] = "The solution converged."
        return True

    if cnt == maxit:
        res['success'] = False
        res['message'] = f"Maximum number of {maxit} iterations reached."
        return True

    if jnp.isnan(eps):
        res['success'] = False
        res['message'] = f"Function returns NaNs"
        return True

    return False


def newton_jax(func, init, maxit=30, tol=1e-8, rtol=None, solver=None, verbose=True, verbose_jac=False):
    """Newton method for root finding of `func` using automatic differenciation with jax. The argument `func` must be jittable with jax. `newton_jax` itself is not jittable, for this use `newton_jax_jit`.

    Parameters
    ----------
    func : callable
        Function f for which f(x)=0 should be found. Is assumed to return a pair (value, jacobian) or (value, jacobian, aux). If not, `val_and_jacfwd` will be applied to the function, in which case the function must be jittable with jax.
    init : array
        Initial values of x
    maxit : int, optional
        Maximum number of iterations
    tol : float, optional
        Required tolerance. Defaults to 1e-8
    solver : callable, optional
        Provide a custom solver `solver(J,f)` for J@x = f. defaults to `jax.numpy.linalg.solve`
    verbose : bool, optional
        Whether to display messages
    verbose_jac : bool, optional
        Whether to supply additional information on the determinant of the jacobian (computationally more costly).

    Returns
    -------
    res: dict
        A dictionary of results similar to the output from scipy.optimize.root
    """

    st = time.time()
    verbose |= verbose_jac
    rtol = rtol or tol

    res = {}
    cnt = 0
    eps_fval = 1e8
    xi = jnp.array(init)

    while True:

        xold = xi.copy()
        cnt += 1
        # evaluate function
        fout = func(xi)

        # remap function if jacobian is not returned
        if not isinstance(fout, tuple):
            func = val_and_jacfwd(func)
            fout = func(xi)

        # unwrap results
        fval, jacval, aux = fout if len(fout) == 3 else (*fout, None)
        # check for convergence or errors
        jac_is_nan = jnp.isnan(jacval.data).any() if isinstance(
            jacval, ssp_csr_array) else jnp.isnan(jacval).any()
        eps = jnp.abs(fval).max()
        if _perform_checks_newton(res, eps, cnt, jac_is_nan, tol, rtol, maxit):
            break

        # be informative
        if verbose and cnt:
            ltime = time.time() - st
            info_str = f'    Iteration {cnt:3d} | max. error {eps:.2e} | lapsed {ltime:3.4f}'
            if verbose_jac:
                jacval = jacval.toarray() if isinstance(
                    jacval, ssp_csr_array) else jacval
                jacdet = jnp.linalg.det(jacval) if (
                    jacval.shape[0] == jacval.shape[1]) else 0
                info_str += f' | det {jacdet:1.5g} | rank {jnp.linalg.matrix_rank(jacval)}/{jacval.shape[0]}'
            print(info_str)

        # assign suitable solver if not given
        if solver is None:
            if isinstance(jacval, ssp_csr_array):
                solver = ssp.linalg.spsolve
            else:
                solver = jax.scipy.linalg.solve
        xi -= solver(jacval, fval)

    jacval = jacval.toarray() if isinstance(
        jacval, (ssp_csr_array, ssp_lil_array)) else jacval

    res['x'], res['niter'] = xi, cnt
    res['fun'], res['jac'] = fval, jacval
    if aux is not None:
        res['aux'] = aux

    if verbose_jac:
        # only calculate determinant if requested
        res['det'] = jnp.linalg.det(jacval) if (
            jacval.shape[0] == jacval.shape[1]) else 0
    else:
        res['det'] = None

    return res
