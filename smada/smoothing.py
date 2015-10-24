#!/usr/bin/env python
"""
Smoothing

This module contains frequentist smoothing methods
"""

# current pylint rating: 7.21/10

import numpy as np


class KernelSmoother(object):
    """Nadayama-Watson Kernel Regression using leave one out pseudo
    cross-validation. Most of the notation is from Wassermann's books"""

    def __init__(self, minh=None, maxh=None, ncv=500, verbose=False):
        """Regression of r(x) ~ y using Nadayama-Watson Kernel Regression

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
        self.X = None
        self.y = None
        self._differences = None
        self.npoints = None
        self.bandwidths = slice(minh, maxh, 1j*ncv)
        self.risks = np.zeros(ncv, 'd')
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
        self.X = X.reshape((-1, 1))
        self.X.shape = (-1, 1)
        self.npoints = X.shape[0]
        self._differences = self.X-self.X.T

        # Now perform cross validation
        _flat_differences = self._differences.ravel()
        _flat_differences = _flat_differences[_flat_differences > 0]
        if self.bandwidths.start is None:
            self.bandwidths.start = np.min(_flat_differences)
        if self.bandwidths.stop is None:
            self.bandwidths.stop = 2*np.max(_flat_differences)
        self.bandwidths = np.mgrid[self.bandwidths]

        if self.verbose:
            print "Starting cross validation (", len(self.risks), "folds )"
        for i in xrange(len(self.bandwidths)):
            if self.verbose:
                print "Fold", i
            self.risks[i] = self._cv_risk(self.bandwidths[i])
        if self.verbose:
            print "Done"

        self.i_opt = np.argmin(self.risks)
        self.h_opt = self.bandwidths[self.i_opt]

        return self

    def _cv_risk(self, bandwidth):
        """Calculate the leave-one-out crossvalidated risk

        This function uses an explicit formula for the leave-one-out
        crossvalidation score as given in Wasserman (2006, p 70)

        Parameters:
            h, float
                scalar bandwidth parameter
        """
        L = self._differences/bandwidth
        L *= L
        L *= -.5
        np.exp(L, L)
        L /= L.sum(1).reshape((-1, 1))
        predicted = np.dot(L, self.y)
        residual = np.sum(((self.y-predicted)/(1-np.diag(L)))**2)/self.npoints
        if not residual == residual:  # Avoid NAN
            return 1e10
        return residual

    def predict(self, X):
        """Evaluate the regression function at the values in X"""
        X.shape = (-1, 1)
        L = self.X.T-X
        L /= self.h_opt
        L *= L
        L *= -.5
        np.exp(L, L)
        L /= L.sum(1).reshape((-1, 1))
        return np.dot(L, self.y).ravel()

    def std_error(self, X):
        """Determine asymptotic standard error"""
        # Get the norm
        L = self.X.T - X
        L /= self.h_opt
        L *= L
        L *= -.5
        np.exp(L, L)
        L /= L.sum(1).reshape((-1, 1))
        L **= 2
        lnorm = L.sum(1)  # not sure about this index

        # Estimate sg
        L = self._differences/self.h_opt
        L *= L
        L *= -.5
        np.exp(L, L)
        L /= L.sum(1).reshape((-1, 1))
        residual = np.dot(L, self.y)
        residual_sd = np.sum((self.y-residual)**2) / (len(self.y)-1)

        return np.sqrt(residual_sd*lnorm).ravel()

    def bootstrap(self, X, nsamples=200):
        """Use bootstrap to estimate standard error

        This is (i) not tested and (ii) very slow.
        """
        residuals = np.zeros((nsamples, len(X)), 'd')
        for i in xrange(nsamples):
            idx = np.random.randint(self.npoints, size=self.npoints)
            L = self.X[idx].T - X
            L /= self.h_opt
            L *= L
            L *= -.5
            np.exp(L, L)
            L /= L.sum(1).reshape((-1, 1))
            residuals[i] = np.dot(L, self.y[idx])
        return residuals
