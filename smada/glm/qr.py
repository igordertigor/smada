#!/usr/bin/env python
"""
Map reduce implementations of QR decompositions
"""
import numpy as np


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


def mapreduce_svd(X_chunks, n=None, map_func=map, reduce_func=reduce):
    R = np.concatenate(map_func(qr_mapped, X_chunks), 0)
    return np.linalg.svd(R)


def lm_solve_qr(data_iter, map_func, reduce_func):
    R = mapreduce_qr(data_iter, map_func=map_func, reduce_func=reduce_func)
    w = np.linalg.solve(R[:-1, :-1], R[:-1, -1])
    return w, R[-1, -1]**2
