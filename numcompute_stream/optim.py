"""
Finite-difference optimisation utilities for NumCompute-Stream.

This module keeps the gradient and Jacobian estimators from the original
NumCompute package. They are retained because finite differences are useful
general numerical tools, even though the main streaming models are tree-based.
"""

import numpy as np


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
    if method not in ("central", "forward"):
        raise ValueError(f"method must be 'central' or 'forward', got '{method}'")

    x = np.asarray(x, dtype=float)
    n = x.size
    perturb = np.eye(n) * h

    if method == "central":
        values = [
            f(x + perturb[i]) - f(x - perturb[i])
            for i in range(n)
        ]
        return np.asarray(values, dtype=float) / (2 * h)

    f0 = f(x)
    values = [
        f(x + perturb[i]) - f0
        for i in range(n)
    ]
    return np.asarray(values, dtype=float) / h


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
    if method not in ("central", "forward"):
        raise ValueError(f"method must be 'central' or 'forward', got '{method}'")

    x = np.asarray(x, dtype=float)
    n = x.size
    perturb = np.eye(n) * h

    if method == "central":
        cols = [
            (np.asarray(F(x + perturb[i])) - np.asarray(F(x - perturb[i]))) / (2 * h)
            for i in range(n)
        ]
    else:
        F0 = np.asarray(F(x))
        cols = [
            (np.asarray(F(x + perturb[i])) - F0) / h
            for i in range(n)
        ]

    return np.column_stack(cols)