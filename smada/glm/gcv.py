#!/usr/bin/env python
"""
Tools for setting regularization parameters based on generalized cross validation
"""

import operator
import functools
import numpy as np

from . import utils


def gcv(R, Qy, B, n, r2=0.):
    """Get generalized cross validation error

    This refers to the following objective:
        $$
        \|y - QRw\|^2 + w^TB^TBw
        $$

    Args:
        R: upper right triangular matrix from QR decomposition of design matrix
        Qy: target vector multiplied by Q from QR decomposition of design matrix
        B: cholesky factor of penalty matrix
        n: total number of observations
        r2: r^2 from projection of y on Q

    Returns:
        generalized cross validation score
    """
    U, D, V = np.linalg.svd(np.r_[R, B])
    d = len(D)
    dl = np.trace(np.dot(U[:d, :d], U[:d, :d].T))
    _, R_ = np.linalg.qr(np.r_[np.c_[R, Qy], np.c_[B, np.zeros(B.shape[0])]])
    score = n*(r2+R_[-1, -1])/(n-dl)**2
    return score


def gcv_as_error(log_th, penalties, getR, nobservations):
    """Call generalized cross validation as function of regularization weights

    Args:
        log_th: log-regularization weights
        penalties: a list of penalty matrices corresponding to the
            regularization weights
        getR: a function that takes the cholesky factor B of a penalty matrix
            and returns an upper triangular matrix R such that there exists an
            orthogonal matrix Q such that QR = np.c_[np.r_[X, B], np.r_[y, 0]].
        nobservations: total number of observations

    Returns:
        generalized cross validation score
    """
    B = utils.block_diag(*map(operator.mul, penalties, np.exp(log_th)))
    R = getR(B)
    return gcv(R[:-1, :-1], R[:-1, -1], B, nobservations, R[-1, -1])


def getR_linear(B, R):
    """Function to return the R factor needed by gcv_as_error

    Note: With given data, get the desired getR function through
        `functools.partial(getR_linear, R=X)`

    Args:
        B: cholesky factor of penalty matrix
        R: R factor of the raw linear problem: np.c_[X,y] = QR. This also works for R=np.c_[X,y].

    Returns:
        R: R factor of the regularized problem with penalty $B^TB$
    """
    return np.linalg.qr(np.r_[R, np.c_[B, np.zeros(B.shape[0])]])[1]


def soptimize(th0, error_func, niter=500, ftol=1e-7):
    """Simplex optimization

    Args:
        th0: starting value
        error_func: function to be minimized
        niter: maximum number of iterations
        ftol: simplex size at which to stop optimizing

    Returns:
        Optimized parameters for which error_func has a local minimum
    """
    k = len(th0)
    if th0 is None:
        th = np.zeros((k+1, k), 'd')
    else:
        th = np.array([th0 for i in xrange(k+1)])
    th[1:] += np.eye(k)
    f = np.array(map(error_func, th))

    al, gm, rho, sg = -1., 2., .5, .5
    for i in xrange(niter):
        ksort = np.argsort(f)
        f = f[ksort]
        th = th[ksort]

        assert f[0] <= f[-1], 'Sorting failed: {}'.format(f)  # check that we have the right convention
        centroid = th[:-1].mean(0)

        reflected, fr = change_point_on_centroid(th[-1], centroid, error_func, al)
        if fr >= f[0] and fr < f[-2]:
            th[-1], f[-1] = reflected, fr
            continue

        if fr < f[0]:
            expanded, fe = change_point_on_centroid(th[-1], centroid, error_func, gm)
            th[-1], f[-1] = (expanded, fe) if fe < fr else (reflected, fr)
            continue

        assert fr >= f[-2], '{} should be larger than {}'.format(fr, f[-2])  # just to make sure
        contracted, fc = change_point_on_centroid(th[-1], centroid, error_func, rho)

        if fc < f[-1]:
            th[-1], f[-1] = contracted, fc
            continue

        res = map(
            functools.partial(change_point_on_centroid, c=centroid, f=error_func, al=sg),
            th[1:])
        th[1:] = [v[0] for v in res]
        f[1:] = [v[1] for v in res]

        if np.var(f) < ftol:
            print 'Convergence after {} iterations'.format(i)
            break
    else:
        print 'No convergence'
    return th[np.argsort(f)[0]]


def change_point_on_centroid(x, c, f, al):
    """Helper function for simplex optimization"""
    x_new = c + al*(x-c)
    return x_new, f(x_new)
