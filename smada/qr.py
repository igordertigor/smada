#!/usr/bin/env python
# TODO: document, combine IRLS into one function
import numpy as np
import time


def qr_mapped(X, n=None):
    """Mapper for QR decomposition

    Args:
        X: matrix chunk
        n: number of rows after which to perform QR decomposition. If X has less
            than n rows, we simply return X. By default, n equals the number of
            columns in X

    Returns:
        X or R, such that QR=X for an orthogonal matrix Q
    """
    return X if X.shape[0] < (n or X.shape[1]) else np.linalg.qr(X)[1]


def qr_reduce(X, Y, n=None):
    """Reducer function for QR decomposition

    For the reduce step, we form a matrix Z such that the top part of Z is equal
    to X and the bottom part is equal to Y. QR decomposition is then applied to
    this combined matrix.

    Args:
        X: matrix chunk
        Y: matrix chunk
        n: number of total rows at which to perform QR decomposition. If Z has
            less than n rows, we simply return XY. By default, n equals the
            number of columns in X(or Y).

    Returns:
        Z or R, such that QR=Z for an orthogonal matrix Q
    """
    return qr_mapped(np.concatenate((X, Y), 0))


def mapreduce_qr(X_chunks, n=None, map_func=map, reduce_func=reduce):
    """Use mapreduce formulation of QR decomposition

    The main strength about this implementation lies in the flexibility provided
    by being able to exchange the map and reduce functions. For example, many
    parallelization libraries provide efficient implementations of the map step
    that could be plugged in to obtain parallel QR decompositions.
    Alternatively, iterator implementations of map and reduce might be used to
    reduce the memory footprint incurred in the computation.

    Args:
        X_blocks: an iterable over matrix chunks
        n: number of rows after which a chunk is transformed
        map_func: the map function that takes a function and an iterable and
            returns the iterable over function values
        reduce_func: the reduce function that takes a function of two arguments
            and an iterable and returns a value constructed by applying the
            function to the first and second elements of the iterable, and then
            repeatedly applying the function to the result and the next element
            from the iterable.

    Returns:
        upper right triangular matrix R such that the concatenation X of all
        matrix chunks can be written as X=QR with an orthogonal matrix Q
    """
    return reduce_func(lambda x, y: qr_reduce(x, y, n),
                       map_func(qr_mapped, X_chunks))


def Xchunked(X, blocksize):
    """Utility function that chunks up a large design matrix

    Note that this may not always be the ideal way to go, as "real" iterators
    might also allow loading chunks of data from external sources, such as data
    bases.

    Args:
        X: full design matrix
        blocksize: number of rows for each matrix chunk

    Yields:
        blocksize rows from the full matrix X
    """
    i = -blocksize
    while i < X.shape[0]-blocksize:
        i += blocksize
        yield X[i:i+blocksize]


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


def example_linear():
    """Example parallel implementation of linear regression"""
    from multiprocessing import Pool
    P = Pool(4)

    N = 5000000  # many observations

    # simulate data
    w = [.2, .1, -.1, .3]
    X = np.ones((N, 5), 'd')
    X[:, 1:4] = np.random.randn(X.shape[0], 3)
    X[:, 4] = np.dot(X[:, :-1], w) + np.random.randn(X.shape[0])

    print 'Linear regression on 4 workers,', N, 'observations, 4 features'
    t0 = time.time()
    R = mapreduce_qr(Xchunked(X, 5000), n=5000, map_func=P.map)
    w_ = np.linalg.solve(R[:-1, :-1], R[:-1, -1])
    print "True regression weights", w
    print "Estimated regression weights", w_
    print 'Execution took', time.time() - t0, 's'


def example_logistic():
    """Example parallel implementation of logistic regression with binary
    response variable"""
    from multiprocessing import Pool
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
    w_ = np.zeros(4, 'd')
    for i in xrange(5):
        print 'Iteration', i
        R = mapreduce_qr(
            weight_iterable(binomial_logistic_family, w_, Xchunked(X, 5000)),
            n=5000,
            map_func=P.map)
        wnew = np.linalg.solve(R[:-1, :-1], R[:-1, -1])
        stepsize = np.sum((w_-wnew)**2)
        w_ = wnew
        print 'Estimated weights', w_
        if stepsize < 1e-7:
            print 'Converged'
            break
    else:
        print 'No convergence'
    print 'Execution took', time.time() - t0, 's'


def example_poisson():
    """Example parallel implementation of poisson regression"""
    from multiprocessing import Pool
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
    w_ = np.zeros(4, 'd')
    for i in xrange(5):
        print 'Iteration', i
        R = mapreduce_qr(
            weight_iterable(poisson_log_family, w_, Xchunked(X, 5000)),
            n=5000,
            map_func=P.map)
        wnew = np.linalg.solve(R[:-1, :-1], R[:-1, -1])
        stepsize = np.sum((w_-wnew)**2)
        w_ = wnew
        print 'Estimated weights', w_
        if stepsize < 1e-7:
            print 'Converged'
            break
    else:
        print 'No convergence'
    print 'Execution took', time.time() - t0, 's'


if __name__ == '__main__':
    print "="*40
    example_linear()
    print "="*40
    example_logistic()
    print "="*40
    example_poisson()
    print "="*40
