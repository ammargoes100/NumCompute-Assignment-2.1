"""
Finite-difference optimisation utilities for NumCompute-Stream.

This module keeps the gradient and Jacobian estimators from the original
NumCompute package. They are retained because finite differences are useful
general numerical tools, even though the main streaming models are tree-based.

Extra validation is added so invalid step sizes, empty inputs, and unexpected
function outputs fail with clearer errors.
"""

import numpy as np


def _validate_method(method):
    """
    Check that the finite difference method is supported.
    """
    if method not in ("central", "forward"):
        raise ValueError(f"method must be 'central' or 'forward', got '{method}'")


def _validate_step_size(h):
    """
    Check that the finite difference step size is positive.
    """
    if not np.isscalar(h) or h <= 0:
        raise ValueError("h must be a positive scalar")


def _as_1d_float_array(x, name="x"):
    """
    Convert input to a non-empty 1D float array.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")

    if x.size == 0:
        raise ValueError(f"{name} cannot be empty")

    return x


def grad(f, x, h=1e-5, method="central"):
    """
    Estimate the gradient of a scalar function using finite differences.

    Parameters
    ----------
    f : callable
        Scalar function f(x) -> float.
    x : array-like of shape (n,)
        Point where the gradient is estimated.
    h : float, default=1e-5
        Step size.
    method : {"central", "forward"}, default="central"
        Finite difference method.

    Returns
    -------
    np.ndarray of shape (n,)
        Estimated gradient.
    """
    _validate_method(method)
    _validate_step_size(h)

    x = _as_1d_float_array(x, name="x")
    n = x.size
    perturb = np.eye(n) * h

    if method == "central":
        values = []

        for i in range(n):
            forward_value = f(x + perturb[i])
            backward_value = f(x - perturb[i])
            values.append(forward_value - backward_value)

        gradient = np.asarray(values, dtype=float) / (2 * h)

    else:
        f0 = f(x)
        values = []

        for i in range(n):
            values.append(f(x + perturb[i]) - f0)

        gradient = np.asarray(values, dtype=float) / h

    if gradient.ndim != 1 or gradient.shape[0] != n:
        raise ValueError("f must return a scalar value")

    return gradient


def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian of a vector-valued function using finite differences.

    Parameters
    ----------
    F : callable
        Vector-valued function F(x) -> array-like of shape (m,).
    x : array-like of shape (n,)
        Point where the Jacobian is estimated.
    h : float, default=1e-5
        Step size.
    method : {"central", "forward"}, default="central"
        Finite difference method.

    Returns
    -------
    np.ndarray of shape (m, n)
        Estimated Jacobian matrix.
    """
    _validate_method(method)
    _validate_step_size(h)

    x = _as_1d_float_array(x, name="x")
    n = x.size
    perturb = np.eye(n) * h

    base_output = np.asarray(F(x), dtype=float)

    if base_output.ndim != 1:
        raise ValueError("F must return a 1D array")

    if base_output.size == 0:
        raise ValueError("F output cannot be empty")

    if method == "central":
        cols = []

        for i in range(n):
            forward_output = np.asarray(F(x + perturb[i]), dtype=float)
            backward_output = np.asarray(F(x - perturb[i]), dtype=float)

            if forward_output.shape != base_output.shape:
                raise ValueError("F output shape changed during evaluation")

            if backward_output.shape != base_output.shape:
                raise ValueError("F output shape changed during evaluation")

            cols.append((forward_output - backward_output) / (2 * h))

    else:
        cols = []

        for i in range(n):
            forward_output = np.asarray(F(x + perturb[i]), dtype=float)

            if forward_output.shape != base_output.shape:
                raise ValueError("F output shape changed during evaluation")

            cols.append((forward_output - base_output) / h)

    return np.column_stack(cols)