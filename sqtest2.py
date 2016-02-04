#!/usr/bin/env python

import numpy as np
from smada import sqpenalty as sqp
import pylab as pl
import seaborn as sns
from sklearn.linear_model import LassoCV, Lasso

n = 200
nlambda = 40
lm = np.mgrid[0:5:1j*nlambda]
nbases = 99

X = np.ones((n, nbases), 'd')
X[:, 1:] = np.random.randn(n, nbases-1)
w = np.random.randn(nbases)
put_zeros = np.random.rand(nbases) < .5  # 50% sparsity
w[put_zeros] = 0

y = np.dot(X, w) + np.random.randn(n)

_, R0 = np.linalg.qr(np.c_[X[:100], y[:100]])
_, R1 = np.linalg.qr(np.c_[X[100:], y[100:]])

e = np.zeros(nlambda, 'd')
B = np.zeros((nlambda, nbases), 'd')
e2 = np.zeros(nlambda, 'd')
B2 = np.zeros((nlambda, nbases), 'd')
for i in xrange(len(lm)):
    b, converged = sqp.estimate_lasso_em(R0[:-1, :-1], R0[:-1, -1], 100*lm[i], R0[-1, -1], niter=50)
    e[i] += np.sum((np.dot(X[100:], b) - y[100:])**2)
    M = Lasso(alpha=lm[i])
    M.fit(X[:100], y[:100])
    e2[i] += np.sum((M.predict(X[100:]) - y[100:])**2)

    b, converged = sqp.estimate_lasso_em(R1[:-1, :-1], R1[:-1, -1], 100*lm[i], R1[-1, -1], niter=50)
    e[i] += np.sum((np.dot(X[:100], b) - y[:100])**2)
    M = Lasso(alpha=lm[i])
    M.fit(X[100:], y[100:])
    e2[i] += np.sum((M.predict(X[:100]) - y[:100])**2)

    B[i] = b
    B2[i] = M.coef_

lm_best = lm[np.argmin(e)]
_, R = np.linalg.qr(np.r_[R0, R1])
b, converged = sqp.estimate_lasso_em(R[:-1, :-1], R[:-1, -1], lm_best, R[-1, -1], niter=50, crit=1e-10)

M = LassoCV(cv=2)
M.fit(X, y)

print np.sum(b == 0)

print converged
for i in xrange(nbases):
    print '{} {:.4f} {:.4f} {:.4f}'.format(i, w[i], b[i], M.coef_[i])

pl.subplot(231)
pl.plot(w)
pl.plot(b)
pl.plot(M.coef_)

pl.subplot(232)
pl.plot(w - b)

pl.subplot(233)
pl.plot(lm, e)
pl.axvline(lm_best)
pl.plot(lm, e2)

pl.subplot(234)
pl.plot(lm, B)
pl.ylim(-.001, .001)

pl.subplot(235)
pl.plot(lm, B2)
pl.ylim(-.001, .001)

pl.show()
