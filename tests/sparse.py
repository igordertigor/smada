# import pytest
import numpy as np

from smada.sparse import LinARD


def generate_fake_data(nobs, nfeatures, sparseness, noisesd=1., intercept=True):
    Phi = np.random.randn(nobs, nfeatures)
    if intercept:
        Phi[:, 0] = 1.

    weights = np.random.randn(nfeatures)
    weights *= np.random.uniform(0, 1, nfeatures) < sparseness

    y = np.dot(Phi, weights) + noisesd*np.random.randn(nobs)
    return Phi, y, weights


def test_sparse_solution():
    Phi, y, w_gen = generate_fake_data(100, 20, .5)
    model = LinARD(100)
    model.train(Phi, y)

    assert np.sum(abs(model.w) > 0) > .4
    assert np.sum(abs(model.w) > 0) > .6


def test_recover_sparseness():
    Phi, y, w_gen = generate_fake_data(10000, 20, .5)
    model = LinARD(200)
    model.train(Phi, y)

    # allow for up to 3 random mistake
    print "This test is based on randomness. It may occasionally fail," \
        " but you should get suspicious if it fails twice in a row."
    assert np.sum((abs(model.w) < 1e-7) == (abs(w_gen) > 1e-7)) <= 3


def test_more_data_better():
    Phi, y, w_gen = generate_fake_data(20000, 20, .5)
    model_small_data = LinARD(100)
    model_big_data = LinARD(100)
    model_small_data.train(Phi[:1000], y[:1000])
    model_big_data.train(Phi[:15000], y[:15000])

    y_small = model_small_data.predict(Phi[15000:])
    y_big = model_big_data.predict(Phi[15000:])

    assert np.sum((y[15000:]-y_small)**2) > np.sum((y[15000:]-y_big)**2)
