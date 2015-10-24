#!/usr/bin/env python
"""
Smoothing

This module contains frequentist smoothing methods
"""

import numpy as np


class KernelSmoother(object):
    """Nadayama-Watson Kernel Regression using leave one out pseudo cross-validation"""

    def __init__(self, minh=None, maxh=None, ncv=500, verbose=False):
        """Estimate regession function r(x) ~ y using Nadayama-Watson Kernel Regression

        Parameters:
            minh, float
                minimum bandwidth to be tested via cross-validation
            maxh, float
                maximum bandwidth to be tested via cross-validation
            ncv, int
                number of bandwidth to be tested via cross-validation
            verbose, bool
                see messages?
        """
        self.x = None
        self.y = None
        self.d = None
        self.n = None
        self.h = slice(minh, maxh, 1j*ncv)
        self.R = np.zeros(ncv, 'd')
        self.i_opt = None
        self.h_opt = None
        self.verbose = verbose

    def train(self, X, y):
        """Train the model

        Parameters:
            X, 1d-array
                array of predictor values
            y, 1d-array
                array of target values
        """
        self.y = y
        self.x = X.reshape((-1, 1))
        self.x.shape = (-1, 1)
        self.n = X.shape[0]
        self.d = self.x-self.x.T

        # Now perform cross validation
        dd = self.d.ravel()
        dd = dd[dd > 0]
        if self.h.start is None:
            self.h.start = np.min(dd)
        if self.h.stop is None:
            self.h.stop = 2*np.max(dd)
        self.h = np.mgrid[self.h]

        if self.verbose:
            print "Starting cross validation (", len(self.R), "folds )"
        for i in xrange(len(self.h)):
            if self.verbose:
                print "Fold", i
            self.R[i] = self.CVrisk(self.h[i])
        if self.verbose:
            print "Done"

        self.i_opt = np.argmin(self.R)
        self.h_opt = self.h[self.i_opt]

        return self

    def CVrisk(self, h):
        """Calculate the leave-one-out crossvalidated risk

        This function uses an explicit formula for the leave-one-out
        crossvalidation score as given in Wasserman (2006, p 70)

        Parameters:
            h, float
                scalar bandwidth parameter
        """
        L = self.d/h
        L *= L
        L *= -.5
        np.exp(L, L)
        L /= L.sum(1).reshape((-1, 1))
        r = np.dot(L, self.y)
        R = np.sum(((self.y-r)/(1-np.diag(L)))**2)/self.n
        if not R == R:  # Avoid NAN
            return 1e10
        return R

    def predict(self, X):
        """Evaluate the regression function at the values in X"""
        X.shape = (-1, 1)
        L = self.x.T-X
        L /= self.h_opt
        L *= L
        L *= -.5
        np.exp(L, L)
        L /= L.sum(1).reshape((-1, 1))
        return np.dot(L, self.y).ravel()

    def se(self, X):
        """Determine asymptotic standard error"""
        # Get the norm
        L = self.x.T - X
        L /= self.h_opt
        L *= L
        L *= -.5
        np.exp(L, L)
        L /= L.sum(1).reshape((-1, 1))
        L **= 2
        lnorm = L.sum(1)  # not sure about this index

        # Estimate sg
        L = self.d/self.h_opt
        L *= L
        L *= -.5
        np.exp(L, L)
        L /= L.sum(1).reshape((-1, 1))
        r = np.dot(L, self.y)
        sg = np.sum((self.y-r)**2) / (len(self.y)-1)

        return np.sqrt(sg*lnorm).ravel()

    def bootstrap(self, x, n=200):
        r = np.zeros((n, len(x)), 'd')
        for i in xrange(n):
            b = np.random.randint(self.n, size=self.n)
            L = self.x[b].T - x
            L /= self.h_opt
            L *= L
            L *= -.5
            np.exp(L, L)
            L /= L.sum(1).reshape((-1, 1))
            r[i] = np.dot(L, self.y[b])
        return r



