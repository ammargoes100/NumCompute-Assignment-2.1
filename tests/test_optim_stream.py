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
def test_grad_rejects_non_positive_step_size():
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        grad(f, x, h=0.0)


def test_grad_rejects_empty_x():
    f = lambda x: np.sum(x ** 2)

    with pytest.raises(ValueError):
        grad(f, np.array([]))


def test_grad_rejects_non_1d_x():
    f = lambda x: np.sum(x ** 2)

    with pytest.raises(ValueError):
        grad(f, np.array([[1.0, 2.0]]))


def test_grad_rejects_vector_output_function():
    f = lambda x: np.array([x[0], x[0] ** 2])
    x = np.array([1.0])

    with pytest.raises(ValueError):
        grad(f, x)


def test_jacobian_rejects_non_positive_step_size():
    F = lambda x: np.array([x[0] ** 2])
    x = np.array([1.0])

    with pytest.raises(ValueError):
        jacobian(F, x, h=-1e-5)


def test_jacobian_rejects_empty_x():
    F = lambda x: np.array([1.0])

    with pytest.raises(ValueError):
        jacobian(F, np.array([]))


def test_jacobian_rejects_non_1d_x():
    F = lambda x: np.array([x[0, 0]])

    with pytest.raises(ValueError):
        jacobian(F, np.array([[1.0, 2.0]]))


def test_jacobian_rejects_scalar_output():
    F = lambda x: x[0] ** 2
    x = np.array([1.0])

    with pytest.raises(ValueError):
        jacobian(F, x)


def test_jacobian_rejects_changing_output_shape():
    def F(x):
        if x[0] > 1.0:
            return np.array([x[0], x[0] ** 2])
        return np.array([x[0]])

    x = np.array([1.0])

    with pytest.raises(ValueError):
        jacobian(F, x, method="forward")