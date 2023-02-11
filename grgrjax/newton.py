# -*- coding: utf-8 -*-

import jax
import time
import jax.numpy as jnp
import scipy.sparse as ssp


def newton_cond_func(carry):
    (xi, eps, cnt), (func, verbose, maxit, tol) = carry
    cond = cnt < maxit
    cond = jnp.logical_and(cond, eps > tol)
    cond = jnp.logical_and(cond, ~jnp.isnan(eps))
    verbose = jnp.logical_and(cnt, verbose)
    jax.debug.callback(callback_func, cnt, eps, verbose=verbose)
    return cond

def newton_body_func(carry):
    (xi, eps, cnt), (func, verbose, maxit, tol) = carry
    xi_old = xi
    f, jac = func(xi)
    xi -= jax.scipy.linalg.solve(jac, f)
    eps = amax(xi-xi_old)
    return (xi, eps, cnt+1), (func, verbose, maxit, tol)

def callback_func(cnt, err, dampening=None, ltime=None, verbose=True):
    mess = f'    Iteration {cnt:3d} | max. error {err:.2e}'
    if dampening is not None:
        mess += f' | dampening {dampening:1.3f}'
    if ltime is not None:
        mess += f' | lapsed {ltime:3.4f}s'
    if verbose:
        print(mess)

@jax.jit
def newton_jax_jit(func, x_init, maxit=30, tol=1e-8, verbose=True):
    """Newton method for root finding using automatic differentiation with jax and running in jitted jax.
    ...

    Parameters
    ----------
    func : callable
        Function returning (y, jac) where f(x)=y=0 should be found and jac is the jacobian. Must be jittable with jax. Could e.g. be the output of jacfwd_and_val.
    x_init : array
        Initial values of x
    maxit : int, optional
        Maximum number of iterations
    tol : float, optional
        Random seed. Defaults to 0

    Returns
    -------
    res: (xopt, (fopt, jacopt), niter, success)
    """
    (xi, eps, cnt), _ = jax.lax.while_loop(newton_cond_func,
                                           newton_body_func, ((x_init, 1., 0), (func, verbose, maxit, tol)))
    return xi, func(xi), cnt, eps > tol

def newton_jax(func, init, jac=None, maxit=30, tol=1e-8, rtol=None, sparse=False, solver=None, func_returns_jac=False, inspect_jac=False, verbose=False, verbose_jac=False):
    """Newton method for root finding using automatic differenciation with jax. The argument `func` must be jittable with jax.

    ...

    Parameters
    ----------
    func : callable
        Function f for which f(x)=0 should be found. Must be jittable with jax
    init : array
        Initial values of x
    jac : callable, optional
        Function that returns the jacobian. If not provided, jax.jacfwd is used
    maxit : int, optional
        Maximum number of iterations
    tol : float, optional
        Random seed. Defaults to 0
    sparse : bool, optional
        Whether to calculate a sparse jacobian. If `true`, and jac is supplied, this should return a sparse matrix
    solver : callable, optional
        Provide a custom solver
    func_returns_jac : bool, optional
        Set to `True` if the function also returns the jacobian.
    inspect_jac : bool, optional
        If `True`, use grgrlib.plots.spy to visualize the jacobian
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
    verbose_jac |= inspect_jac
    verbose |= verbose_jac
    rtol = rtol or tol

    if jac is None and not func_returns_jac:
        if sparse:
            def jac(x): return ssp.csr_array(jax.jacfwd(func)(x))
        else:
            jac = jax.jacfwd(func)

    if solver is None:
        if sparse:
            solver = ssp.linalg.spsolve
        else:
            solver = jax.scipy.linalg.solve

    res = {}
    cnt = 0
    xi = jnp.array(init)

    while True:

        xold = xi.copy()
        jacold = jacval.copy() if cnt else None
        cnt += 1

        if func_returns_jac:
            fout = func(xi)
            fval, jacval, aux = fout if len(fout) == 3 else (*fout, None)
            if sparse and not isinstance(jacval, ssp._arrays.csr_array):
                jacval = ssp.csr_array(jacval)
        else:
            fout, jacval = func(xi), jac(xi)
            fval, aux = fout if len(fout) == 2 else (fout, None)

        jac_is_nan = jnp.isnan(jacval.data) if isinstance(
            jacval, ssp._arrays.csr_array) else jnp.isnan(jacval)
        if jac_is_nan.any():
            res['success'] = False
            res['message'] = "The Jacobian contains `NaN`s."
            jacval = jacold if jacold is not None else jacval
            break

        eps_fval = jnp.abs(fval).max()
        if eps_fval < tol:
            res['success'] = True
            res['message'] = "The solution converged."
            break

        xi -= solver(jacval, fval)
        eps = jnp.abs(xi - xold).max()

        if verbose:
            ltime = time.time() - st
            info_str = f'    Iteration {cnt:3d} | max. error {eps:.2e} | lapsed {ltime:3.4f}'
            if verbose_jac:
                jacval = jacval.toarray() if sparse else jacval
                jacdet = jnp.linalg.det(jacval) if (
                    jacval.shape[0] == jacval.shape[1]) else 0
                info_str += f' | det {jacdet:1.5g} | rank {jnp.linalg.matrix_rank(jacval)}/{jacval.shape[0]}'
                if inspect_jac:
                    spy(jacval)

            print(info_str)

        if cnt == maxit:
            res['success'] = False
            res['message'] = f"Maximum number of {maxit} iterations reached."
            break

        if eps < rtol:
            res['success'] = True
            res['message'] = "The solution converged."
            break

        if jnp.isnan(eps):
            res['success'] = False
            res['message'] = f"Function returns 'NaN's"
            break

    jacval = jacval.toarray() if isinstance(
        jacval, (ssp._arrays.csr_array, ssp._arrays.lil_array)) else jacval

    res['x'], res['niter'] = xi, cnt
    res['fun'], res['jac'] = fval, jacval
    if aux is not None:
        res['aux'] = aux

    if verbose_jac:
        # only calculate determinant if requested
        res['det'] = jnp.linalg.det(jacval) if (jacval.shape[0] == jacval.shape[1]) else 0
    else:
        res['det'] = None

    return res
