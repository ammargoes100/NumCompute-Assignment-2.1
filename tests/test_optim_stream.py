"""
Tests for retained finite-difference optimisation utilities.
"""

import numpy as np
import pytest

from numcompute_stream.optim import grad, jacobian


def test_grad_central_on_quadratic():
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0, 3.0])

    result = grad(f, x, method="central")
    expected = 2 * x

    np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_grad_forward_on_quadratic():
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0, 3.0])

    result = grad(f, x, method="forward")
    expected = 2 * x

    np.testing.assert_array_almost_equal(result, expected, decimal=3)


def test_grad_raises_on_bad_method():
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        grad(f, x, method="backward")


def test_jacobian_values_central():
    F = lambda x: np.array([x[0] ** 2, x[1] ** 2])
    x = np.array([1.0, 3.0])

    result = jacobian(F, x, method="central")
    expected = np.array([
        [2.0, 0.0],
        [0.0, 6.0],
    ])

    np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_jacobian_forward_method():
    F = lambda x: np.array([x[0] * x[1], x[0] + x[1]])
    x = np.array([2.0, 3.0])

    result = jacobian(F, x, method="forward")
    expected = np.array([
        [3.0, 2.0],
        [1.0, 1.0],
    ])

    np.testing.assert_array_almost_equal(result, expected, decimal=3)


def test_jacobian_raises_on_bad_method():
    F = lambda x: np.array([x[0] ** 2])
    x = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        jacobian(F, x, method="backward")


def test_jacobian_output_shape():
    F = lambda x: np.array([x[0] ** 2 + x[1], x[2] * x[0]])
    x = np.array([1.0, 2.0, 3.0])

    result = jacobian(F, x)

    assert result.shape == (2, 3)