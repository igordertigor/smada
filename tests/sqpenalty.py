import pytest
import numpy as np
import smada.sqpenalty as sqp


@pytest.fixture(params=[True, False])
def compress(request):
    return request.param


def test_QR(compress):
    """Test online QR decomposition

    We do this my making sure that recovered parameters and residuals are independent of weather the data are observed
    in one go or in multiple chunks.
    """
    np.random.seed(0)
    X = np.random.randn(40, 10)
    y = np.random.randn(40)

    QR1 = sqp.OnlineQR(compress=compress)
    QR2 = sqp.OnlineQR(compress=compress)

    QR1.update(X, y)
    QR2.update(X[:20], y[:20])
    QR2.update(X[20:], y[20:])

    R1, f1, r1 = QR1.get()
    R2, f2, r2 = QR2.get()
    b1 = np.linalg.solve(R1, f1)
    b2 = np.linalg.solve(R2, f2)

    assert abs(r1-r2) < 1e-5
    assert np.all(abs(b1-b2) < 1e-5)


def test_consistency_with_fixed_regularizer():
    """As we collect more data, the estimated parameters should approximate generating values"""
    np.random.seed(0)
    b = np.ones(4, 'd')
    X = np.random.randn(100, 4)
    y = np.random.randn(100) + np.dot(X, b)

    S = np.eye(4)

    b20, r2 = sqp.solve_penalized(X[:20], y[:20], S)
    b100, r2 = sqp.solve_penalized(X, y, S)

    assert np.linalg.norm(b20-1) > np.linalg.norm(b100-1)


def test_ridge_residual():
    """As we increase the regularization strength two things should happen:
        1. The training error should increase
        2. The norm of the parameter vector should decrease
    """
    np.random.seed(0)
    b = np.ones(4, 'd')
    X = np.random.randn(40, 4)
    y = np.random.randn(40) + np.dot(X, b)
    S = np.eye(4)

    b1, r1 = sqp.solve_penalized(X, y, S)
    b4, r4 = sqp.solve_penalized(X, y, 4*S)

    assert r1 < r4
    assert np.linalg.norm(b1) > np.linalg.norm(b4)


@pytest.fixture(params=[
    sqp.NaturalSplineBasis(np.mgrid[0:1:20j]),
    sqp.B_SplineBasis(np.mgrid[0:1:20j])])
def basis(request):
    return request.param


def test_function_approximation(basis):
    """Using more data with the spline bases should actually result in better function estimates"""
    x = np.random.rand(100, 1)
    y = np.cos(4*x) + (x-.5)**2 + np.random.randn(100, 1)*.2

    dmat1, reg1, constr1 = sqp.setup_additive(x[:50], [basis])
    dmat2, reg2, constr2 = sqp.setup_additive(x, [basis])
    model1 = sqp.SquaredPenaltyModel(reg1, constr1)
    model2 = sqp.SquaredPenaltyModel(reg2, constr2)

    model1.fit(dmat1, y[:50])
    model2.fit(dmat2, y)

    x2 = np.mgrid[0:1:100j]
    dpre, _, _ = sqp.setup_additive(x2.reshape((-1, 1)), [basis])

    y1 = model1.predict(dpre)
    y2 = model2.predict(dpre)
    ytrue = np.cos(4*x2) + (x2-.5)**2

    # Compare (numerical) integral L2-norms
    assert np.sqrt(np.trapz((y2-ytrue)**2, x2)) < np.sqrt(np.trapz((y1-ytrue)**2, x2))


def test_constraints():
    basis = sqp.NaturalSplineBasis(np.mgrid[0:1:20j])
    x = np.random.rand(100, 1)
    y = np.cos(4*x) + (x-.5)**2 + np.random.randn(100, 1)*.2

    dmat, reg, constr = sqp.setup_additive(x, [basis])

    Z = sqp.constraint_transformation(constr)
    assert abs(np.dot(constr, Z)).max() < 1e-5
