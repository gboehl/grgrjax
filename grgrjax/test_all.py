# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from . import newton_jax, newton_jax_jit, val_and_jacfwd


def f(x):
    res = x.copy()
    res = res.at[0].set(jnp.sin(x[0]))
    res = res.at[1].set(jnp.log(x[1]))
    return res


def solver(jval, fval):
    """A solver to solve indetermined problems.
    """
    return jnp.linalg.pinv(jval) @ fval


vaj_f = jax.tree_util.Partial(val_and_jacfwd(f))
x0 = jnp.ones(2)*.1


def test0():
    r0 = newton_jax(f, x0)
    assert r0['success']


def test1():
    r0 = newton_jax(vaj_f, x0)
    assert r0['success']


def test2():
    r0 = newton_jax(vaj_f, x0, verbose_jac=True)
    assert r0['success']


def test3():
    r0 = newton_jax(vaj_f, x0, solver=solver)
    assert r0['success']


def test4():
    r0 = newton_jax_jit(vaj_f, x0)
    assert not r0[-1]
