#!/usr/bin/env python

import numpy as np
import pylab as pl
import seaborn as sns
import smada.sqpenalty as sqp


def f(x):
    return -(np.exp(.1*x)-.4*x**2 + .1*x**3)
n = 1000

# Create test data
basis = sqp.NaturalSplineBasis(np.mgrid[0:5:40j])
x = 5*np.random.rand(n)
y_true = f(x)
y_obs = y_true + .4*np.random.randn(n)

dmat, reg, constr = sqp.setup_additive(x.reshape((-1, 1)), [basis])

QR = sqp.OnlineQR()
QR.update(dmat, y_obs)

model = sqp.SquaredPenaltyModel(reg, constr)
model.fit(*QR.get(), n=n)

sns.set_style('ticks')

_x = np.mgrid[0:5:100j]
X, _, _ = sqp.setup_additive(_x.reshape((-1, 1)), [basis])
_f = model.predict(X)

pl.subplot(131)
pl.scatter(x, y_obs, label='data')
pl.plot(_x, f(_x), label=r'$f$')
pl.plot(_x, _f, label=r'$\hat{f}$')
pl.title('A) Fitted function')
pl.legend(loc='lower left')
pl.xlabel(r'$x$')
pl.ylabel(r'$f(x)$')

pl.subplot(132)
pl.plot(model.b)
pl.title('B) Model coefficients')
pl.xlabel('index')
pl.ylabel('weight')

pl.subplot(133)
pl.plot(model.eta_grid, model.gcv_grid)
pl.axhline(model.score)
pl.title('C) Cross validation results')
pl.xlabel(r'$\eta=\log(\theta)$')
pl.ylabel(r'Generalized cross validation score')
pl.text(.05, .9, r'$\hat\eta=$({:.3})'.format(float(model.eta)),
        transform=pl.gca().transAxes)

sns.despine()
pl.tight_layout()

pl.show()
