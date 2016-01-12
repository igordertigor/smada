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
