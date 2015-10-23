#!/usr/bin/env python

import numpy as np
import logging

LOGGER = logging.getLogger('smada.sparse')
logging.basicConfig(level=logging.INFO)


class LinARD(object):
    """automatic relevance determination in linear model

    ARD is an empirical Bayes method that optimizes a diagonal prior to optimize
    marginal likelihood. Practically, this is typically quite close to
    optimizing an l0 penalty, i.e. maximizing sparseness.
    """
    # runtime: oct 23, 2015: 3.3866109848

    def __init__(self, niter=100):
        """
        Parameters:
            t, 1d-array
                vector of target values
            Phi, 2d-array
                matrix of feature values
        """
        self.niter = niter
        self.beta = .0001    # shouldn't this be a parameter
        self.alpha = None
        self.w = None
        # self.included = np.ones(self.alpha.shape[0],'bool')
        self.delta = 1e10
        self.al_thres = 20

    def estimate_normal(self, al, X, y):
        """Estimate normal distribution parameters for posterior of weights

        Parameters:
            al, 1d-array
                vector of prior precisions
            Phi, 2d-array
                feature matrix

        Returns:
            m, Sg, Lm
            mean vector, covariance matrix, precision matrix
        """
        Lm = np.diag(np.array(al).ravel()) + self.beta*X.T*X
        Sg = Lm.I
        m = self.beta * Sg*X.T*y

        return m, Sg, Lm

    def get_gm(self, al, Sg):
        """Get responsibilities

        Parameters:
            al, 1d-array
                vector of prior precisions
            Sg, 2d-array
                covariance matrix

        Returns:
            gm
            vector of responsibilities
        """
        gm = 1.-np.array(al).ravel()*np.diag(Sg.A).ravel()
        return gm

    def update(self, X, y):
        """Update all parameters"""
        al = self.alpha[self.included]
        al_old = al.copy()
        X_ = X[:, self.included]
        m, Sg, Lm = self.estimate_normal(al, X_, y)
        gm = self.get_gm(al, Sg)
        al_new = gm/m.A.ravel()**2
        res = np.sum((y-X_*m).A**2)
        bt_new = X.shape[0]-np.sum(gm)
        bt_new /= res
        k = 0
        for i in xrange(X.shape[1]):
            if self.alpha[i] < self.al_thres:
                self.alpha[i] = al_new[k]
                self.included[i] = True
                self.w[i] = m[k]
                k += 1
            if self.alpha[i] >= self.al_thres:
                self.alpha[i] = self.al_thres
                self.included[i] = False
                self.w[i] = 0
        if np.sum(abs(al_new-al_old)) < 1e-7 and abs(bt_new-self.beta) < 1e-7:
            self.beta = bt_new
            return 1
        else:
            self.beta = bt_new
            return 0

    def initialize(self, X, y):
        """Initialize the algorithm"""
        m, Sg, Lm = self.estimate_normal(self.alpha, X, y)
        self.included = np.array(abs(m) > 1e-7).ravel()
        self.alpha[np.logical_not(self.included)] = self.al_thres

    def train(self, X, y):
        """Train the model using niter iterations"""
        # self.alpha = np.matrix(np.random.randn(X.shape[1])**2).T
        X = np.matrix(X)
        y = np.matrix(y).T
        self.alpha = .1*np.ones(X.shape[1])
        self.initialize(X, y)
        self.w = np.zeros(X.shape[1])
        for i in xrange(self.niter):
            if self.update(X, y):
                break
            if (i % 50) == 0:
                al = self.alpha[self.included]
                # al_old = al.copy()
                X_ = X[:, self.included]
                self.m, self.Sg, self.Lm = self.estimate_normal(al, X_, y)
                f = X_*self.m
        else:
            LOGGER.warn("No convergence for smada.sparse.LinARD")
        al = self.alpha[self.included]
        # al_old = al.copy()
        X_ = X[:, self.included]
        self.m, self.Sg, self.Lm = self.estimate_normal(al, X_, y)

    def predict(self, Phi):
        phi = Phi[:, self.included]
        return np.dot(phi, self.m.A.ravel())


def measure_runtime():
    import timeit
    Phi = np.random.randn(100, 10)
    Phi[:, 0] = 1.
    w = [0., 1., 1., 0., -1., -1., 0., 0., 1., -1.]
    y = np.dot(Phi, w) + 1*np.random.randn(100)

    setup = """from numpy import array
from smada.sparse import LinARD
y = {}
Phi = {}"""

    time = timeit.timeit('LinARD(y, Phi).train(100)',
                         setup=setup.format(repr(y), repr(Phi)), number=100)
    LOGGER.info("Measured runtime {}".format(time))
    return time


if __name__ == "__main__":
    measure_runtime()
