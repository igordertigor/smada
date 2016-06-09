#!/usr/bin/env python

import time
import numpy as np

from multiprocessing import Pool

from smada.glm import glm, qr, utils


def example_linear():
    """Example parallel implementation of linear regression"""
    P = Pool(4)

    N = 5000000  # many observations

    # simulate data
    w = [.2, .1, -.1, .3]
    X = np.ones((N, 5), 'd')
    X[:, 1:4] = np.random.randn(X.shape[0], 3)
    X[:, 4] = np.dot(X[:, :-1], w) + np.random.randn(X.shape[0])

    print 'Linear regression on 4 workers,', N, 'observations, 4 features'
    t0 = time.time()
    R = qr.mapreduce_qr(utils.Xchunked(X, 5000), n=5000, map_func=P.map)
    w_ = np.linalg.solve(R[:-1, :-1], R[:-1, -1])
    print "True regression weights", w
    print "Estimated regression weights", w_
    print 'Execution took', time.time() - t0, 's'


def example_logistic():
    """Example parallel implementation of logistic regression with binary
    response variable"""
    P = Pool(4)

    N = 5000000  # many observations

    # Simulate data
    w = [.2, .1, -.1, .3]
    X = np.ones((N, 5), 'd')
    X[:, 1:4] = np.random.randn(X.shape[0], 3)
    X[:, 4] = 1./(1+np.exp(-np.dot(X[:, :-1], w)))
    X[:, 4] = np.random.rand(N) < X[:, 4]

    print 'Logistic regression on 4 workers,', N, 'observations, 4 features'
    print 'True generating weights', w
    t0 = time.time()

    w_est, r2, converged = glm.estimate_glm(
        X, glm.binomial_logistic_family,  # penalty=np.eye(4),
        niter=5, map_func=P.map)
    print 'Estimated weights', w_est
    print 'Final R2', r2
    print 'Converged' if converged else 'Not converged'

    print 'Execution took', time.time() - t0, 's'


def example_poisson():
    """Example parallel implementation of poisson regression"""
    P = Pool(4)

    N = 5000000  # Many observations

    # Simulate data
    w = [.2, .1, -.1, .3]
    X = np.ones((N, 5), 'd')
    X[:, 1:4] = np.random.randn(X.shape[0], 3)
    X[:, 4] = np.exp(np.dot(X[:, :-1], w))
    X[:, 4] = np.random.poisson(X[:, 4])

    print 'Poisson regression on 4 workers,', N, 'observations, 4 features'
    print 'True generating weights', w
    t0 = time.time()
    w_est, r2, converged = glm.estimate_glm(
        X, glm.poisson_log_family, niter=5, map_func=P.map)
    print 'Estimated weights', w_est
    print 'Final R2', r2
    print 'Converged' if converged else 'Not converged'

    print 'Execution took', time.time() - t0, 's'


def example_logistic_counts():
    """Example parallel implementation of logistic regression with binary
    response variable"""
    P = Pool(4)

    N = 5000000  # many observations

    # Simulate data
    w = [.2, .1, -.1, .3]
    X = np.ones((N, 5), 'd')
    X[:, 1:4] = np.random.randn(X.shape[0], 3)
    X[:, 4] = 1./(1+np.exp(-np.dot(X[:, :-1], w)))
    X[:, 4] = np.random.rand(N) < X[:, 4]
    counts = np.random.poisson(10, size=(N,))

    print 'Logistic regression on counts on 4 workers,', N, 'observations, 4 features'
    print 'True generating weights', w
    t0 = time.time()

    w_est, r2, converged = glm.estimate_glm(
        (X, counts), glm.binomial_logistic_family,  # penalty=np.eye(4),
        niter=5, map_func=P.map)
    print 'Estimated weights', w_est
    print 'Final R2', r2
    print 'Converged' if converged else 'Not converged'

    print 'Execution took', time.time() - t0, 's'


if __name__ == '__main__':
    print "="*40
    example_linear()
    print "="*40
    example_logistic()
    print "="*40
    example_poisson()
    print "="*40
    example_logistic_counts()
    print "="*40
