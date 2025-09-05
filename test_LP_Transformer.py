from LP_Transformer import BasicTransformer

from LP import LP
import numpy as np


def test_JustEqualities():
    """
    min 2 * x + 3 * y
        s.t. 1*x + 2*y = 3
             4*x + 5*y = 6
    """
    A = np.array([[1, 2], [4, 5]])
    b = np.array([3, 6])
    c = np.array([2, 3])
    basicLP = LP(c, A, b)
    trans = BasicTransformer(basicLP)
    transLP = trans.Transform()

    newA = transLP.A_eq
    newb = transLP.b_eq
    newc = transLP.c

    expectedA = np.hstack([A, -A])
    expectedc = np.hstack([c, -c])
    assert np.allclose(newA, expectedA)
    assert np.allclose(newb, b)
    assert np.allclose(newc, expectedc)


def test_2():
    """
    min 2 * x + 3 * y
        s.t. 1*x + 2*y = 3
             4*x + 5*y = 6
             -1.1 <= x <= 2.1
    """
    A = np.array([[1, 2], [4, 5]])
    b = np.array([3, 6])
    c = np.array([2, 3])
    basicLP = LP(c, A, b, lb=np.array([-1.1, np.nan]), ub=np.array([2.1, np.nan]))
    trans = BasicTransformer(basicLP)
    transLP = trans.Transform()

    newA = transLP.A_eq
    newb = transLP.b_eq
    newc = transLP.c

    expectedA = np.vstack(
        [
            np.hstack([A, -A, np.zeros((2, 2))]),
            np.array([[-1, 0, 1, 0, 1, 0], [1, 0, -1, 0, 0, 1]]),
        ]
    )
    expectedb = np.hstack([b, np.array([1.1, 2.1])])

    excpectedc = np.hstack([c, -c, np.zeros(2)])

    assert np.allclose(newA, expectedA)
    assert np.allclose(newb, expectedb)
    assert np.allclose(newc, excpectedc)
