#!/usr/bin/env python
"""
Basis expansions
"""

import functools
import numpy as np


def bspline(x, knots, i, degree):
    if degree == 0:
        B = np.logical_and(x >= knots[i], x < knots[i+1]).astype('d')
    else:
        if abs(knots[degree+i] - knots[i]) < 1e-7:
            alpha1 = 0
        else:
            alpha1 = (x - knots[i])/(knots[degree+i] - knots[i])

        if abs(knots[i+degree+1] - knots[i+1]) < 1e-7:
            alpha2 = 0
        else:
            alpha2 = (knots[i+degree+1] - x)/(knots[i+degree+1] - knots[i+1])

        B = alpha1*bspline(x, knots, i, degree-1)
        B += alpha2*bspline(x, knots, i+1, degree-1)

    return B


def bs_expand(x, knots, degree, map_func=map):
    K = len(knots) + degree - 1  # ninterior knots + degree + 1
    knots_ = np.concatenate((
        [knots[0]]*(degree+1),
        knots[1:-1],
        [knots[-1]]*(degree+1)))
    return np.array(map_func(
        functools.partial(bspline, x, knots_, degree=degree), range(K))).T


def bs_penalty(knots, degree):
    K = len(knots) + degree - 1
    C = np.eye(K)
    C -= 2*np.eye(K, k=1)
    C += np.eye(K, k=2)
    return C


if __name__ == "__main__":
    knots = np.mgrid[0:1:10j]
    x = np.mgrid[0:1:100j]

    import pylab as pl
    pl.plot(x, bs_expand(x, knots, 3))
    pl.savefig('test.pdf')
