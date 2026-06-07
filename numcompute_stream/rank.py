"""
Ranking utilities for NumCompute-Stream.

This module keeps the ranking and percentile helpers from the original
NumCompute package. They are retained because they are useful for basic
numerical analysis, tie handling, and percentile calculations.

Example print statements from the original file are intentionally removed so
the module can be imported without producing unwanted output.
"""

import numbers
import numpy as np


def _validate_1d_non_empty(values, function_name):
    """
    Convert input to an array and check that it is a non-empty 1D array.
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(f"{function_name} expects a 1D array")

    if values.size == 0:
        raise ValueError(f"{function_name} cannot operate on an empty array")

    return values


def _check_no_nan(values, function_name):
    """
    Reject NaN values for numeric arrays.
    """
    if np.issubdtype(values.dtype, np.number) and np.isnan(values).any():
        raise ValueError(f"{function_name} does not support NaN values")


def rank(values, method="average"):
    """
    Rank a 1D array with tie handling.

    Parameters
    ----------
    values : array-like of shape (n,)
        Values to rank.
    method : {"average", "dense", "ordinal"}, default="average"
        Ranking method used for ties.

    Returns
    -------
    np.ndarray
        Rank values starting at 1.
    """
    values = _validate_1d_non_empty(values, "rank")
    _check_no_nan(values, "rank")

    if method not in {"average", "dense", "ordinal"}:
        raise ValueError("method must be one of {'average', 'dense', 'ordinal'}")

    n = values.size
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]

    if method == "ordinal":
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        return ranks

    group_start = np.empty(n, dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_values[1:] != sorted_values[:-1]

    starts = np.flatnonzero(group_start)
    ends = np.r_[starts[1:], n]

    sorted_ranks = np.empty(n, dtype=float if method == "average" else int)

    if method == "average":
        for start, end in zip(starts, ends):
            average_rank = (start + 1 + end) / 2.0
            sorted_ranks[start:end] = average_rank

    else:
        dense_rank_values = np.arange(1, starts.size + 1)

        for dense_rank, start, end in zip(dense_rank_values, starts, ends):
            sorted_ranks[start:end] = dense_rank

    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks

    return ranks


def percentile(values, q, interpolation="linear"):
    """
    Compute the q-th percentile of a 1D numeric array.

    Parameters
    ----------
    values : array-like of shape (n,)
        Input values.
    q : int or float
        Percentile in the range [0, 100].
    interpolation : {"linear", "lower", "higher", "midpoint"}, default="linear"
        Interpolation method used when q falls between two values.

    Returns
    -------
    scalar
        Percentile value.
    """
    values = _validate_1d_non_empty(values, "percentile")
    _check_no_nan(values, "percentile")

    if not isinstance(q, numbers.Real) or isinstance(q, bool):
        raise TypeError("q must be a numeric value")

    if q < 0 or q > 100:
        raise ValueError("q must be in the range [0, 100]")

    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError(
            "interpolation must be one of {'linear', 'lower', 'higher', 'midpoint'}"
        )

    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("percentile expects numeric values")

    sorted_values = np.sort(values.astype(float), kind="stable")
    n = sorted_values.size

    if n == 1:
        return sorted_values[0]

    position = (q / 100.0) * (n - 1)
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))

    if lower_index == upper_index:
        return sorted_values[lower_index]

    if interpolation == "lower":
        return sorted_values[lower_index]

    if interpolation == "higher":
        return sorted_values[upper_index]

    if interpolation == "midpoint":
        return (sorted_values[lower_index] + sorted_values[upper_index]) / 2.0

    weight = position - lower_index

    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )