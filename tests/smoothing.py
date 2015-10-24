import numpy as np

from smada.smoothing import KernelSmoother


def generate_fake_data(xmin, xmax, nobs):
    x = np.random.uniform(xmin, xmax, nobs)
    x.sort()
    generating_rate = 4*np.exp(-.1*(x-10)**2)
    # y = np.random.poisson(generating_rate)
    y = np.random.randn(nobs) + generating_rate
    return x, y, generating_rate


def test_se():
    # np.random.seed(0)
    x, y, generating_rate = generate_fake_data(0, 20, 100)

    K = KernelSmoother(ncv=20, minh=.1, maxh=5)
    K.train(x, y)

    estimated_rate = K.predict(x)
    std_err = K.se(x)

    x.shape = (-1,)

    # Standard errors are with respect to a smoothed version of the target
    # function. Apply smoothing here and interpolate linearly
    x_ = np.mgrid[0:20:100j]
    h = np.exp(-.5*(x_-10)**2/K.h_opt)
    h /= np.sum(h)
    gr = 4*np.exp(-.1*(x_-10)**2)
    g = np.convolve(gr, h, 'same')
    smoothed_targets = np.interp(x, x_, g)

    # import pylab as pl
    # pl.fill_between(x, estimated_rate+3*std_err, estimated_rate-3*std_err, alpha=.5)
    # pl.plot(x, smoothed_targets)
    # pl.show()

    print "This test is based on randomness. It may occasionally fail," \
        " but you should get suspicious if it fails twice in a row."
    error_indicator = (abs(estimated_rate - smoothed_targets) > 3*std_err).astype('d')
    # We allow for one contiguous segment of error
    nsteps = np.sum(np.diff(error_indicator) > 0)
    assert nsteps < 2

    # The error segment should not be larger than 50 elements (that's quite
    # generous)
    assert np.sum(error_indicator) < 50


def test_smaller_bandwith_with_more_data():
    n = 100

    x, y, generating_rate = generate_fake_data(0, 20, n)

    K1 = KernelSmoother(ncv=40, minh=.1, maxh=5).train(x[:.1*n], y[:.1*n])
    K2 = KernelSmoother(ncv=40, minh=.1, maxh=5).train(x, y)

    print "This test is based on randomness. It may occasionally fail," \
        " but you should get suspicious if it fails twice in a row."
    assert K2.h_opt <= K1.h_opt

if __name__ == "__main__":
    test_se()
