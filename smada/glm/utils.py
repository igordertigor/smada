import itertools
import numpy as np


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


def get_data_iter(data, nchunks=100):
    if isinstance(data, np.ndarray):
        def data_iter():
            return Xchunked(data, int(np.ceil(data.shape[0]/nchunks)))
    elif isinstance(data, tuple):
        data, counts = data
        chunk_size = int(np.ceil(data.shape[0]/nchunks))

        def data_iter():
            return itertools.izip(Xchunked(data, chunk_size),
                                  Xchunked(counts, chunk_size))
    elif getattr(data, '__iter__', False):
        def data_iter():
            return itertools.tee(data)[1]
    else:
        raise ValueError('Unknown input type {}'.format(type(data)))
    return data_iter


def block_diag(*matrices):
    if matrices == ():
        matrices = ([],)
    matrices = [np.atleast_2d(m) for m in matrices]
    bad_sizes = [k for k in range(len(matrices)) if matrices[k].ndim > 2]
    if bad_sizes:
        raise ValueError("Arguments in the following positions have dimensions"
                         "greater than 2: {}".format(bad_sizes))

    shapes = np.array([mat.shape for mat in matrices])
    out = np.zeros(np.sum(shapes, axis=0), dtype=matrices[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r+rr, c:c+cc] = matrices[i]
        r += rr
        c += cc
    return out
