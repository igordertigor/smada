#!/usr/bin/env python
"""
Implements methods for fitting linear models with regularizers that are quadratic in the parameters and application of
these methods to additive models based on penalized regression splines.

The implemented methods are largely from the following papers:
    Wood (2004): Stable and efficient multiple smoothing parameter estimation for generalized additive models. JASA, 99,
    673-686.
    Wood, Goude, Shaw (2015): Generalized additive models for large data sets. Appl Statist, 64(1), 139-155.
    Wood & Augustin (2002): GAMs with integrated model selection using penalized regression splines and applications to
    environmental modeling. Ecological Modelling, 157(2-3) 157-177.
"""

import numpy as np
from scipy.linalg import solve_triangular
import scipy.optimize as opt


class NaturalSplineBasis(object):
    """The natural spline basis

    For knots x*_i, i=1, ..., q this is given by
    .. math::
        N_i(x) = |x-x*_i|^3, i=1, ..., q
        N_{q+1}(x) = x, N_{q+2}(x) = 1

    Natural splines are the preferred choice for penalized regression. In particular, it can be shown that the best
    approximating smooth function (smooth in terms of energy of the second derivative) to n points (x_i, y_i) can be
    expressed in the natural spline basis with knots chosen to be the points x_i.

    When fitting a natural spline basis two constraints need to be satisfied:
        The spline should average to 0
        The spline should continue linearly beyond the borders of the data.
    """
    def __init__(self, knots):
        """:param knots: the sequence of spline knots"""
        self.q = len(knots)
        self.knots = knots

    def expand(self, predictor):
        """Exand predictor in the natural spline basis

        :param predictor: vector of predictor values

        :returns: array with shape (Nrecords,q+2)
        """
        return np.concatenate((abs(predictor.reshape((-1, 1)) - self.knots.reshape((1, -1)))**3,
                               np.ones(predictor.shape, 'd').reshape((-1, 1)),
                               predictor.reshape((-1, 1))), 1)

    def penalty(self):
        """Create matrix representation of smoothness penalty for this basis"""
        S = np.zeros((self.q+2, self.q+2), 'd')
        # # As far as I understand, the following should be the penalty, but it isn't positive semi-definite
        # S[:-2, :-2] = self.func.outer(self.knots, self.knots)
        # # Weaker alternative: sum of squared derivative at knots, not integral
        for x in self.knots:
            S[:-2, :-2] += np.outer(np.abs(x-self.knots), np.abs(x-self.knots))

        # # Other attempt using the integral
        # l = np.min(self.knots)
        # u = np.max(self.knots)
        # S[:-2, :-2] += np.diag(((u-self.knots)**3 - (l-self.knots)**3)/3.)
        # for i in xrange(self.q):
        #     for j in xrange(i+1, self.q):
        #         S[i, j] = 2*((u**3-l**3)/3. +
        #                      self.knots[i]*self.knots[j]*(u-l) -
        #                      (u*u-l*l)*0.5*(self.knots[i]+self.knots[j]))
        #         S[j, i] = S[i, j]

        return S

    def constraints(self):
        """Build constraint matrix for this basis"""
        C = np.zeros((2, self.q+2), 'd')
        C[0, :self.q] = 1
        C[1, :self.q] = self.knots
        return C

    def __len__(self):
        """Number of basis functions"""
        return self.q+2
    func = np.frompyfunc(lambda x, y: np.abs(x-y)**3, 2, 1)


class B_SplineBasis(object):
    """B-spline basis

    B-splines span the same space as natural splines and are in addition localized. Yet, the roughness penalty usually
    used for B-splines differs from the penalty term used for natural splines and B-spline smooths are typically
    visually more wiggly than smooths obtained using natural splines.
    """
    def __init__(self, knots, degree=3):
        """
        :param knots: an array of B-spline knots. Note that duplicate knots are removed from the array of knots and then
        the boundary knots are repeated.
        :param degree: degree of the underlying polynomial.
        """
        self.q = len(knots)
        if degree < 1:
            raise ValueError('Spline degree must be at least 1')
        self.degree = degree

        self.knots = np.sort(np.unique(knots))
        self.K = len(self.knots) + degree - 1  # ninterior knots + degree + 1
        self.knots = np.concatenate(([self.knots[0]]*(degree+1), self.knots[1:-1], [self.knots[-1]]*(degree+1)))

    def expand(self, predictor):
        """Expand a predictor in the B-spline basis

        :param predictor: a 1d array containing predictor values

        :returns: a 2d array of shape (Nrecords, Nknots) that can be used as design matrix
        """
        B = np.zeros((len(predictor), self.K), 'd')
        for j in xrange(self.K):
            B[:, j] = self._B(predictor, self.degree, j, self.knots)
        endknots = predictor == self.knots[-1]
        if np.any(endknots):
            B[endknots, -1] = 1
        return B

    def penalty(self):
        """Smoothness penalty typically used for B-splines"""
        S = 2*np.eye(self.K)
        S[0, 0] = 1
        S[-1, -1] = 1
        S -= np.diag(np.ones(self.K-1), 1)
        S -= np.diag(np.ones(self.K-1), -1)
        return S

    def constraints(self):
        """Constraint matrix"""
        return np.ones((1, self.K), 'd')

    def __len__(self):
        """Number of basis functions"""
        return self.K

    @staticmethod
    def _B(x, degree, i, knots):
        """A helper method to recursively evaluate B-splines"""
        if degree == 0:
            B = np.logical_and(x > knots[i], x < knots[i+1]).astype('d')
        else:
            if abs(knots[degree+i] - knots[i]) < 1e-7:
                alpha1 = 0
            else:
                alpha1 = (x - knots[i])/(knots[degree+i] - knots[i])

            if abs(knots[i+degree+1] - knots[i+1]) < 1e-7:
                alpha2 = 0
            else:
                alpha2 = (knots[i+degree+1] - x)/(knots[i+degree+1] - knots[i+1])

            B = alpha1*B_SplineBasis._B(x, degree-1, i, knots)
            B += alpha2*B_SplineBasis._B(x, degree-1, i+1, knots)

        return B


def setup_additive(predictors, bases):
    """Create setup for an additive model
    :param predictors: matrix with shape (Nrecords, Npredictors)
    :param bases: list of Npredictors basis objects

    :returns:
        - dmat  design matrix
        - reg   list of regularizers
        - constr constraint matrix
    """
    nrecords, npredictors = predictors.shape
    assert len(bases) == npredictors

    # Parametric part of the design matrix
    dmat = [np.ones((nrecords, 1), 'd')]
    reg = []
    _constr = []

    # Collect matrix fragments
    for i in range(npredictors):
        X = bases[i].expand(predictors[:, i])
        C = np.r_[bases[i].constraints(), X.sum(0).reshape((1, -1))]   # is it smart to have the second constraint here?
        dmat.append(X)
        reg.append(bases[i].penalty())
        _constr.append(C)

    # Put it all together
    dmat = np.concatenate(dmat, 1)
    constr = []
    i0 = 1
    for i in xrange(npredictors):
        i1 = i0 + len(bases[i])
        C = np.zeros((_constr[i].shape[0], dmat.shape[1]), 'd')
        C[:, i0:i1] = _constr[i]
        constr.append(C)

        S = np.zeros((dmat.shape[1], dmat.shape[1]), 'd')
        S[i0:i1, i0:i1] = reg[i]
        reg[i] = S

        i0 = i1
    constr = np.concatenate(constr, 0)

    return dmat, reg, constr


class OnlineQR(object):
    """Online updating version regression relevant QR parameters

    We can solve the least squares problem Xb = y using the QR decomposition of a new matrix A=[X,y].
    Specifically let A=QR, then the regression weights b solve R[:-1, :-1]*b=R[:-1, -1] and can be determined by
    backward substitution. Furthermore R[-1, -1]**2 is the residual of this model.

    Wood et al (2015) describe a way to parallelize regression problems using the QR decomposition.
    They note that if X=QR for some X=np.r_[X1, X2], with X1=Q1*R1 and X2=Q2*R2, then R12 such that
    Q12*R12 = np.r_[R1, R2] is the same as the original R.

    This object abstracts Wood et al's method. Calls to update() accumulate more and more data into the QR
    decomposition. Calls to get() return the regression relevant parameters R, Qy, and r**2, such that
    R*b=Qy and the sum of squares is r**2.
    """

    def __init__(self, compress=True):
        """
        :param compress: boolean indicating weather or not to re-compress after each call to update() by performing a QR
        decomposition and storing only the R matrix. Note that compress=True requires more computations but results in a
        considerably smaller memory footprint.
        """
        self.compress = compress
        self.__mat = []

    def update(self, X, y):
        """Update the internal QR decomposition parameters

        :param X: design matrix
        :param y: target vector
        """
        if self.compress:
            self.__mat.append(np.c_[X, y])
            self.__mat = [np.linalg.qr(np.concatenate(self.__mat, 0))[1]]
        else:
            self.__mat.append(np.linalg.qr(np.c_[X, y])[1])

    def get(self):
        """Get regression relevant parameters

        :returns:
            - R     upper triangular matrix of the regression problem
            - Qy    transformed target values for the regression problem
            - r2    residual sum of squares
        """
        R = np.linalg.qr(np.concatenate(self.__mat, 0))[1]
        return R[:-1, :-1], R[:-1, -1], R[-1, -1]**2


class SquaredPenaltyModel(object):
    """Linear model with square penalty on the parameters"""

    def __init__(self, penalties, constraints=None, eta0=None):
        """
        :param penalties: a list of matrices representing the bilinear kernel of the penalty terms.
        :param constraints: a matrix of hard constraints C of the form Cb=0, where b is the parameter vector.
        :param eta0: starting value for the search if the regularization strength
        """
        self.n = 0
        self.b = None
        self.eta0 = eta0
        self.eta = None
        self.theta = None
        self.score = None
        self.penalties = penalties
        self.constraints = constraints
        self.eta_grid = None
        self.gcv_grid = None

    def fit(self, R, Qy, r2=0., n=None):
        """Fit the model by optimizating regularization and then determining the regression weights

        :param R: design matrix _or_ R factor of a QR decomposition of the design matrix
        :param Qy: target variable _or_ transpose of Q factor applied to target variable
        :param r2: basic residual sum of squares. This is relevant when using the QR input arguments to compensate for
            the residuals originating from the projection. Yet, this is only relevant if we want to compare gcv scores
            between models.
        :param n: number of records. Again, this is only relevant when using the QR input arguments. Otherwise, we
            simply use the number of rows in the design matrix.
        """
        S = self.penalties
        C = self.constraints
        if n is None:
            n = R.shape[0]
        if C is not None:
            Z, S, R, Qy, r2 = apply_constraints(C, R, Qy, S, r2)

        # do a coarse grid search (1d)
        if self.eta0 is None:
            eta = np.linspace(-4, 5, 20)
            self.eta_grid = eta
            self.gcv_grid = np.zeros(20, 'd')
            g = np.inf
            eta0 = 0.
            args = (R, Qy, S, n, r2)
            for i, eta_ in enumerate(eta):
                g_ = gcv([eta_]*len(S), *args)
                if g_ < g:
                    g = g_
                    eta0 = eta_
                self.gcv_grid[i] = g_
        else:
            eta0 = self.eta0

        if isinstance(eta0, float):
            eta0 = [eta0]*len(S)

        # Now run optimizer
        eta_est = opt.fmin(gcv, eta0, args=args)

        # And get the actual parameters
        b = get_basic_definitions(np.exp(eta_est), R, Qy, S)[-2]

        if C is not None:
            b = np.dot(Z, b)

        # Store in object
        self.n = n
        self.b = b
        self.eta0 = eta0
        self.eta = eta_est
        self.theta = np.exp(eta_est)
        self.score = gcv(eta_est, *args)

    def predict(self, X):
        """Perform a prediction

        :param X: design matrix for which to perform a prediction
        """
        return np.dot(X, self.b)


def constraint_transformation(C):
    """Get the projection matrix Z such that XZa~y satisfies the constraints

    :param C: matrix of linear constraints (i.e. Cb=0, for a parameter vector b)
    """
    C_ = np.zeros((C.shape[1], C.shape[1]), 'd')
    C_[:C.shape[0], :] = C
    Q, R = np.linalg.qr(C_.T, 'full')
    Z = Q.T[:, C.shape[0]:]
    return Z


def get_B(th, S):
    """Get matrix square root of total weighted penalty matrix

    :param th: array of regularization parameters
    :param S_i: list of component penalty matrices
    """
    SS = .1*np.eye(S[0].shape[0])
    for th_, S_ in zip(th, S):
        SS += th_*S_
    return np.linalg.cholesky(SS)


def svd_params(R, B):
    """Get SVD of augmented R matrix

    :param R: upper right triangle matrix from QR decomposition of X
    :param B: square root of total weighted penalty matrix
    """
    U, D, V = np.linalg.svd(np.r_[R, B])
    U1 = U[:len(D), :len(D)]
    return U1, D, V


def solve_penalized(R, Qy, B):
    """Solve penalized regression problem

    :param R: design matrix _or_ R
    :param Qy: target vector _or_ Q.T*y
    :param B: square root of total penalty matrix

    :returns:
        - b     regression weights
        - r2    residual sum of squares
    """
    M = np.c_[np.r_[R, B], np.r_[Qy, np.zeros(B.shape[0], 'd')]]
    q, r = np.linalg.qr(M)
    b = solve_triangular(r[:-1, :-1], r[:-1, -1])
    r2 = np.sum((np.dot(R, b) - Qy)**2)  # Correct residual
    return b, r2


def get_basic_definitions(th, R, Qy, S):
    """Get basic definitions needed for model fitting"""
    B = get_B(th, S)
    U1, D, V = svd_params(R, B)
    dl = np.trace(np.dot(U1, U1.T))
    b, r2 = solve_penalized(R, Qy, B)
    return dl, b, abs(r2)


def gcv(eta, R, Qy, S, n, r2=0.):
    """Evaluate generalized cross validation score

    :param eta: log-regularization strengths
    :param R: design matrix _or_ R
    :param Qy: target vector _or_ Q.T*y
    :param S: list of penalty matrices
    :param n: total number of observations
    """
    th = np.exp(eta)
    dl, b, _r2 = get_basic_definitions(th, R, Qy, S)
    score = n*(_r2+r2)/(n-dl)**2
    return score


def apply_constraints(C, R, f, S, r2=0.):
    """Apply the constraint transformation"""
    Z = constraint_transformation(C)
    S = [np.dot(Z.T, np.dot(Si, Z)) for Si in S]
    Q, R = np.linalg.qr(np.c_[np.dot(R, Z), f])
    f = R[:-1, -1]
    r2 += R[-1, -1]
    R = R[:-1, :-1]
    return Z, S, R, f, r2
