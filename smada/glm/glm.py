#!/usr/bin/env python

import functools
import numpy as np

from . import qr


def estimate_glm(data, link_family, penalty=None, niter=5, xtol=1e-7, map_func=map):
    """Use IRLS to estimate parameters of a generalized linear model

    Args:
        data: a data matrix or an iterator over data chunks.
        link_family: a function that combines link function and exponential
            family. See `normal_identity_family`, `binomial_logistic_family`,
            and `poisson_log_family` for examples.
        penalty: if this is a matrix Q, there is an additional penalty w.TQ.TQw on
            the estimation
        niter: number of reweighted least squares iterations
        xtol: final desired stepsize
        map_func: potential drop in replacement for `map`. This allows to drop
            in parallel implementations of map, resulting in considerably faster
            runtime.

    Returns:
        w: regression weight vector for the model
        r2: (weighted) sum of squares (useful for e.g. generalized cross
            validation and model testing)
        converged: boolean flag to indicate if the iteration converged
    """
    if isinstance(data, np.ndarray):
        def data_iter():
            return qr.Xchunked(data, int(np.ceil(data.shape[0]/10)))
    elif getattr(data, '__iter__', False):
        def data_iter():
            return data.copy()

    if penalty is None:
        reduce_func = reduce
    else:
        def reduce_func(f, seq):
            return reduce(f, seq, np.c_[penalty, np.zeros(penalty.shape[0], 'd')])

    # Initialize with linear model
    R = qr.mapreduce_qr(data_iter(), map_func=map_func, reduce_func=reduce_func)
    w = np.linalg.solve(R[:-1, :-1], R[:-1, -1])

    # IRLS
    for i in range(niter):
        R = qr.mapreduce_qr(
            weight_iterable(link_family, w, data_iter()),
            map_func=map_func,
            reduce_func=reduce_func
        )
        w_ = np.linalg.solve(R[:-1, :-1], R[:-1, -1])
        change = np.sum((w_-w)**2)
        w = w_
        if change < 1e-7:
            converged = True
            break
    else:
        converged = False

    return w, R[-1, -1], converged


def normal_identity_family(eta, y):
    """GLM family variables for normal identity model

    Args:
        eta (array): the product Xb of a generalized linear model
        y (array): the array of target values

    Returns:
        mu, dmu/deta, deta/dmu, var(Y)
    """
    return y, eta, 1., 1., 1.


def binomial_logistic_family(eta, y):
    """GLM family variables for binomial logistic regression model

    Args:
        eta (array): the product Xb of a generalized linear model
        y (array|tuple): the target variable. If this is an array, it is assumed
            to be the vector of class labels (0-1 coding). If this is a tuple,
            it is assumed to consist of two arrays, one containing the observed
            fracton of successes and one containing the number of observations.

    Returns:
        mu, dmu/deta, deta/dmu, var(Y)

    Raises:
        ValueError: if y is an invalid type
    """
    if isinstance(y, tuple):
        y, n = y
    elif isinstance(y, np.ndarray):
        n = 1.
        assert np.sum((np.sort(np.unique(y)) - [0, 1])**2) < 1e-7
    else:
        raise ValueError('unknown target format')
    mu = 1./(1+np.exp(-eta))
    dmudeta = mu*(1-mu)
    vary = n * dmudeta
    detadmu = 1./np.clip(dmudeta, 1e-7, np.inf)  # Clip to avoid nan
    return y, mu, dmudeta, detadmu, vary


def poisson_log_family(eta, y):
    """GLM family variables for Poisson logistic regression model

    Args:
        eta (array): the product Xb of a generalized linear model
        y (array): the target variable counts

    Returns:
        mu, dmu/deta, deta/dmu, var(Y)
    """
    mu = np.exp(eta)
    detadmu = 1./mu
    return y, mu, mu, detadmu, mu


def weight_iterable(family, beta, iterable):
    """Weight chunks returned from iterable for reweighted least squares

    Args:
        family: a function that takes arguments eta=X*beta and responses y and
            returns the GLM relevant variables y, mu, dmu/deta, deta/dmu,
            var(y). See for example `normal_identity_family`,
            `binomial_logistic_family`, or `poisson_log_family` for
            implementations that could be used here.
        beta: current regression weights
        iterable: iterable over data chunks

    Yields:
        weighted matrix chunks
    """
    for X in iterable:
        if isinstance(X, tuple):
            # This allows for multi-observation binomial targets
            X, target = X
        else:
            X, target = X[:, :-1], X[:, -1]
        eta = np.dot(X, beta)
        target, mu, dmudeta, detadmu, vary = family(eta, target)
        z = eta + (target - mu)*detadmu
        # Weights here are square roots of what the weights would be in regular
        # glm contexts
        weights = dmudeta/np.clip(np.sqrt(vary), 1e-7, np.inf)
        yield weights.reshape((-1, 1))*np.c_[X, z]
