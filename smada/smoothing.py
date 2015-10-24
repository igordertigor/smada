#!/usr/bin/env python
"""
Smoothing

This module contains methods for one dimensional smoothing
"""

# current pylint score: 8.41/10 (main problem is L-matrix in Kernel Smoother

import numpy as np
from scipy import optimize

from smada import kernels


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


class GaussianProcessRegression(object):
    """Gaussian process regression for smoothing one dimensional data"""

    def __init__(self, kernel=None, noise_precision=1., optimize_bw=False):
        """initialize"""
        self.noise_precision = noise_precision
        self.optimize_bandwidth = optimize_bw
        self.X = None
        self.y = None
        if isinstance(kernel, kernels.Kernel):
            self.kernel = kernel
        else:
            self.kernel = kernels.GaussKernel(1.)
        self.covar_kernel = None
        self.covar_kernel_inv = None

    def _get_kernel_matrix(self):
        """Calculate the covariance matrix K(Xi, Xj)"""
        nobservations = self.X.shape[0]
        kernel_matrix = np.zeros((nobservations, nobservations), 'd')
        for i in xrange(nobservations):
            for j in xrange(i, nobservations):
                kernel_matrix[i, j] = self.kernel(self.X[i, :], self.X[j, :])
                kernel_matrix[j, i] = kernel_matrix[i, j]
            kernel_matrix[i, i] += 1./self.noise_precision**2
        return np.matrix(kernel_matrix)

    def _get_kernel_vector(self, new_x):
        """Calculate K(x, Xi)

        Parameters:
            x, array
                new locations to predict at
        """
        nobservations = self.X.shape[0]
        k = np.zeros((nobservations,))
        for i in xrange(nobservations):
            k[i] = self.kernel(new_x, self.X[i, :])
        return np.matrix(k).T

    def predict(self, X):
        """Predict at new locations X

        Parameters:
            X, array
                new locations to predict at

        Returns:
            (mean, standard deviation) as a function of X
        """
        nobservations = X.shape[0]
        mean = np.zeros(nobservations, 'd')
        std_dev = np.zeros(nobservations, 'd')
        for i in xrange(nobservations):
            kernel_vector = self._get_kernel_vector(X[i, :])
            total_var = self.kernel(X[i, :], X[i, :]) + 1./self.noise_precision
            kernel_vec_times_inv_covar = kernel_vector.T * self.covar_kernel_inv
            mean[i] = kernel_vec_times_inv_covar * self.y
            std_dev[i] = total_var - kernel_vec_times_inv_covar * kernel_vector
        return mean, std_dev

    def _neglikelihood(self, theta, nobservations):
        """Negative log-likelihood with respect to hyperparameters

        Parameters:
            theta, array
                parameters of the kernel + one for the data precision

        Returns:
            - log likelihood
        """
        theta = np.ravel(np.array(theta))
        self.noise_precision = theta[-1]
        self.kernel.theta = abs(theta[:-1])
        kernel_matrix = self._get_kernel_matrix()
        loglikeli = - np.log(np.linalg.det(kernel_matrix))
        if np.isnan(loglikeli):
            raise ValueError('Invalid likelihood: NaN')
        loglikeli -= self.y.T*kernel_matrix.I*self.y
        loglikeli -= nobservations*np.log(2*np.pi)
        loglikeli *= 0.5
        return -loglikeli

    def train(self, X, y):
        """Train model in predictor values X and targets y"""
        self.X = np.matrix(X)
        self.y = np.matrix(y)
        self.y.shape = (-1, 1)
        self.covar_kernel = self._get_kernel_matrix()
        self.covar_kernel_inv = self.covar_kernel.I

        if self.optimize_bandwidth:
            nobservations = self.X.shape[0]

            theta0 = np.zeros(self.kernel.nprm+1, 'd')
            theta0[:-1] = self.kernel.theta
            theta0[-1] = 1.
            # CG doesn't work here -- takes forever
            theta = optimize.fmin(self._neglikelihood, theta0,
                                  args=(nobservations,), disp=0)
            self.kernel.theta, self.noise_precision = theta[:-1], theta[-1]

        return self
