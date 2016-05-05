"""
Simple example for a poisson GAM with a single smooth term
"""
import numpy as np
import pylab as pl

from smada.glm import expand, glm


def f(x):
    return (.5+5*(1-np.exp(-.5*x**2))*np.sin(2.5*x)*np.exp(-.5*x**2))**2


def example_simple_additive():
    x = np.arange(0, 100, dtype='d')/20
    fx = f(x)
    y = np.random.poisson(fx)

    knots = np.mgrid[0:5:20j]
    X = np.c_[np.ones(x.shape), x, expand.bs_expand(x, knots, 3)]
    B = expand.block_diag(np.eye(2), expand.bs_penalty(knots, 3))

    # We don't yet have proper cross validation to estimate optimal smoothness.
    # Just estimate 3 different smoothness levels
    w0, r, converged = glm.estimate_glm(
        np.c_[X, y], glm.poisson_log_family, .1*B)
    w1, r, converged = glm.estimate_glm(
        np.c_[X, y], glm.poisson_log_family, .01*B)
    w2, r, converged = glm.estimate_glm(
        np.c_[X, y], glm.poisson_log_family, B)

    f0 = np.exp(np.dot(X, w0))
    f1 = np.exp(np.dot(X, w1))
    f2 = np.exp(np.dot(X, w2))

    pl.plot(x, fx)
    pl.plot(x, y, 'o')
    pl.plot(x, f0)
    pl.plot(x, f1)
    pl.plot(x, f2)

    pl.savefig('test.pdf')


if __name__ == '__main__':
    example_simple_additive()
