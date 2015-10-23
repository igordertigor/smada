#!/usr/bin/env python
"""
Sparse and sparseness promoting models.
"""

# current pylint score: 9.52/10

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
        self.coef_ = None
        self.posterior_parameters = None
        self.active_set = None
        # self.included = np.ones(self.alpha.shape[0],'bool')
        self.al_thres = 20

    def estimate_normal(self, alpha, X, y):
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
        Lm = np.diag(np.array(alpha).ravel()) + self.beta*X.T*X
        Sg = Lm.I
        m = self.beta * Sg*X.T*y

        return m, Sg, Lm

    @staticmethod
    def _get_gamma(alpha, Sg):
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
        gamma = 1.-np.array(alpha).ravel()*np.diag(Sg.A).ravel()
        return gamma

    def update(self, X, y):
        """Update all parameters"""
        al_active = self.alpha[self.active_set]
        al_old = al_active.copy()
        X_active = X[:, self.active_set]
        m, Sg, _ = self.estimate_normal(al_active, X_active, y)
        gamma = self._get_gamma(al_active, Sg)
        al_new = gamma/m.A.ravel()**2
        res = np.sum((y-X_active*m).A**2)
        bt_new = X.shape[0]-np.sum(gamma)
        bt_new /= res
        k = 0
        for i in xrange(X.shape[1]):
            if self.alpha[i] < self.al_thres:
                self.alpha[i] = al_new[k]
                self.active_set[i] = True
                self.coef_[i] = m[k]
                k += 1
            if self.alpha[i] >= self.al_thres:
                self.alpha[i] = self.al_thres
                self.active_set[i] = False
                self.coef_[i] = 0
        if np.sum(abs(al_new-al_old)) < 1e-7 and abs(bt_new-self.beta) < 1e-7:
            self.beta = bt_new
            return 1
        else:
            self.beta = bt_new
            return 0

    def initialize(self, X, y):
        """Initialize the algorithm"""
        m, _, _ = self.estimate_normal(self.alpha, X, y)
        self.active_set = np.array(abs(m) > 1e-7).ravel()
        self.alpha[np.logical_not(self.active_set)] = self.al_thres

    def train(self, X, y):
        """Train the model using niter iterations"""
        # self.alpha = np.matrix(np.random.randn(X.shape[1])**2).T
        X = np.matrix(X)
        y = np.matrix(y).T
        self.alpha = .1*np.ones(X.shape[1])
        self.initialize(X, y)
        self.coef_ = np.zeros(X.shape[1])
        for i in xrange(self.niter):
            if self.update(X, y):
                break
            if (i % 50) == 0:
                al_active = self.alpha[self.active_set]
                # al_old = al.copy()
                X_active = X[:, self.active_set]
        else:
            LOGGER.warn("No convergence for smada.sparse.LinARD")
        al_active = self.alpha[self.active_set]
        # al_old = al.copy()
        X_active = X[:, self.active_set]
        m, Sg, Lm = self.estimate_normal(al_active, X_active, y)
        self.posterior_parameters = {
            'mean': m,
            'covariance': Sg,
            'precision': Lm,
        }

    def predict(self, X):
        """Predict for a feature matrix X"""
        X_active = X[:, self.active_set]
        return np.dot(X_active, np.array(self.coef_[self.active_set]).ravel())


def measure_runtime():
    """Measure runtime (currently only of LinARD"""
    import timeit
    X = np.random.randn(100, 10)
    X[:, 0] = 1.
    weights = [0., 1., 1., 0., -1., -1., 0., 0., 1., -1.]
    y = np.dot(X, weights) + 1*np.random.randn(100)

    setup = """from numpy import array
from smada.sparse import LinARD
y = {}
X = {}"""

    time = timeit.timeit('LinARD(y, X).train(100)',
                         setup=setup.format(repr(y), repr(X)), number=100)
    LOGGER.info("Measured runtime {}".format(time))
    return time


if __name__ == "__main__":
    measure_runtime()
