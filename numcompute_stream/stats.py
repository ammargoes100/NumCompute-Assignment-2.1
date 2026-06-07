"""
Descriptive statistics utilities for NumCompute-Stream.

This module builds on the statistics helpers from the original NumCompute
package. The batch functions are retained because they are useful for analysing
complete datasets and individual incoming chunks.

The original WelfordStats class is also retained as a lightweight scalar
running-statistics helper.
"""

import numpy as np


def _validate_array(data):
    """
    Check that data is a non-empty NumPy array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a NumPy array, got {type(data).__name__}")

    if data.size == 0:
        raise ValueError("data must not be empty")


def mean(data, axis=None):
    """
    Compute mean while ignoring NaN values.
    """
    _validate_array(data)

    with np.errstate(all="ignore"):
        return np.nanmean(data, axis=axis)


def median(data, axis=None):
    """
    Compute median while ignoring NaN values.
    """
    _validate_array(data)

    if np.all(np.isnan(data)):
        return np.nan

    with np.errstate(all="ignore"):
        return np.nanmedian(data, axis=axis)


def std(data, axis=None):
    """
    Compute standard deviation while ignoring NaN values.
    """
    _validate_array(data)

    with np.errstate(all="ignore"):
        return np.nanstd(data, axis=axis)


def minimum(data, axis=None):
    """
    Compute minimum while ignoring NaN values.
    """
    _validate_array(data)

    with np.errstate(all="ignore"):
        return np.nanmin(data, axis=axis)


def maximum(data, axis=None):
    """
    Compute maximum while ignoring NaN values.
    """
    _validate_array(data)

    with np.errstate(all="ignore"):
        return np.nanmax(data, axis=axis)


def histogram(data, n_bins=10):
    """
    Compute histogram while ignoring NaN values.
    """
    _validate_array(data)

    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")

    valid_data = data.flatten()
    valid_data = valid_data[~np.isnan(valid_data)]

    if valid_data.size == 0:
        return np.zeros(n_bins, dtype=int), np.array([np.nan] * (n_bins + 1))

    return np.histogram(valid_data, bins=n_bins)


def quantile(data, q, axis=None):
    """
    Compute quantiles while ignoring NaN values.
    """
    _validate_array(data)

    q_array = np.atleast_1d(np.asarray(q, dtype=float))

    if np.any(q_array < 0) or np.any(q_array > 1):
        raise ValueError(f"q values must be between 0 and 1, got {q}")

    with np.errstate(all="ignore"):
        return np.nanquantile(data, q, axis=axis)


class WelfordStats:
    """
    Incremental mean and variance for a scalar stream.
    """

    def __init__(self):
        self.n_samples = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, new_value):
        """
        Update running statistics with one scalar value.

        NaN values are ignored.
        """
        if np.isnan(new_value):
            return self

        self.n_samples += 1

        delta_before = new_value - self._mean
        self._mean += delta_before / self.n_samples
        delta_after = new_value - self._mean
        self._M2 += delta_before * delta_after

        return self

    def mean(self):
        """
        Return current running mean.
        """
        if self.n_samples == 0:
            raise ValueError("No values added yet. Call update() first.")

        return self._mean

    def variance(self):
        """
        Return current population variance.
        """
        if self.n_samples == 0:
            raise ValueError("No values added yet. Call update() first.")

        if self.n_samples == 1:
            return 0.0

        return self._M2 / self.n_samples

    def std(self):
        """
        Return current population standard deviation.
        """
        return np.sqrt(self.variance())

    def reset(self):
        """
        Reset tracker to initial state.
        """
        self.n_samples = 0
        self._mean = 0.0
        self._M2 = 0.0

        return self