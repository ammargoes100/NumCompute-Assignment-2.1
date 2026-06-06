"""
Tests for optim.py
"""
import numpy as np
import pytest
from numcompute.optim import grad, jacobian


def test_grad_central_on_quadratic():
    # f(x) = x^2, gradient should be 2x
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0, 3.0])
    result = grad(f, x, method='central')
    expected = 2 * x
    np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_grad_forward_on_quadratic():
    # same function but using forward differences, slightly less accurate
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0, 3.0])
    result = grad(f, x, method='forward')
    expected = 2 * x
    np.testing.assert_array_almost_equal(result, expected, decimal=3)


def test_grad_raises_on_bad_method():
    # should raise ValueError if method is not central or forward
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        grad(f, x, method='backward')

def test_jacobian_values_central():
    # F(x) = [x0^2, x1^2], J = diag([2*x0, 2*x1])
    F = lambda x: np.array([x[0] ** 2, x[1] ** 2])
    x = np.array([1.0, 3.0])
    result = jacobian(F, x, method='central')
    expected = np.array([[2.0, 0.0], [0.0, 6.0]])
    np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_jacobian_forward_method():
    F = lambda x: np.array([x[0] * x[1], x[0] + x[1]])
    x = np.array([2.0, 3.0])
    result = jacobian(F, x, method='forward')
    expected = np.array([[3.0, 2.0], [1.0, 1.0]])
    np.testing.assert_array_almost_equal(result, expected, decimal=3)


def test_jacobian_raises_on_bad_method():
    F = lambda x: np.array([x[0] ** 2])
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        jacobian(F, x, method='backward')
        
def test_jacobian_output_shape():
    # F: R^3 -> R^2, jacobian should be (2, 3)
    F = lambda x: np.array([x[0] ** 2 + x[1], x[2] * x[0]])
    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(F, x)
    assert result.shape == (2, 3)
