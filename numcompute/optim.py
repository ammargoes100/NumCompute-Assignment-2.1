"""
Module: optim.py
Description: Finite-difference gradient and Jacobian estimation for scalar
             and vector-valued functions. Supports forward and central difference
             methods. Part of the NumCompute toolkit.
"""

import numpy as np


def grad(f, x, h=1e-5, method='central'):
    """
    Estimate the gradient of a scalar function using finite differences.

    Parameters
    f      : callable, f(x) -> float
    x      : np.ndarray, shape (n,)
    h      : float, optional (default=1e-5)
    method : {'central', 'forward'}, optional (default='central')
             central is O(h^2) accurate, forward is O(h)

    Returns
    np.ndarray, shape (n,)

    Raises
    ValueError
        If method is not 'central' or 'forward'.

    Time complexity  : O(n) function evaluations
    Space complexity : O(n)
    """
    if method not in ('central', 'forward'):
        raise ValueError(f"method must be 'central' or 'forward', got '{method}'")

    x = np.asarray(x, dtype=float)
    n = x.size
    perturb = np.eye(n) * h

    if method == 'central':
        return np.array(
            [f(x + perturb[i]) - f(x - perturb[i]) for i in range(n)]
        ) / (2 * h)

    f0 = f(x)
    return np.array(
        [f(x + perturb[i]) - f0 for i in range(n)]
    ) / h


def jacobian(F, x, h=1e-5, method='central'):
    """
    Estimate the Jacobian of a vector-valued function using finite differences.

    Parameters
    F      : callable, F(x) -> np.ndarray of shape (m,)
    x      : np.ndarray, shape (n,)
    h      : float, optional (default=1e-5)
    method : {'central', 'forward'}, optional (default='central')
             central is O(h^2) accurate, forward is O(h)

    Returns
    np.ndarray, shape (m, n)
        Entry [i, j] = dF_i / dx_j

    Raises
    ValueError
        If method is not 'central' or 'forward'.

    Time complexity  : O(n) function evaluations
    Space complexity : O(m * n)
    """
    if method not in ('central', 'forward'):
        raise ValueError(f"method must be 'central' or 'forward', got '{method}'")

    x = np.asarray(x, dtype=float)
    n = x.size
    perturb = np.eye(n) * h

    if method == 'central':
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