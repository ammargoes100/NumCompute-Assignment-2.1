"""
Descriptive statistics utilities for NumCompute-Stream.

This module builds on the statistics helpers from the original NumCompute
package. The batch functions are retained because they are useful for analysing
complete datasets and individual incoming chunks.

Streaming statistics are added so feature-wise mean and variance can be updated
as new chunks arrive.
"""

import numbers
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

    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
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
        if not isinstance(new_value, numbers.Real):
            raise TypeError("new_value must be numeric")

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


class StreamingStats:
    """
    Track feature-wise running mean and variance over data chunks.

    This class is useful when data arrives in batches and the full dataset does
    not need to be stored in memory.
    """

    def __init__(self):
        self.reset()

    def update_stats(self, X_chunk):
        """
        Update running feature-wise statistics using one chunk.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)

        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)

        if X_chunk.ndim != 2:
            raise ValueError("X_chunk must be a 1D or 2D array")

        if X_chunk.shape[0] == 0:
            raise ValueError("X_chunk must contain at least one sample")

        if self.n_features_in_ is None:
            self.n_features_in_ = X_chunk.shape[1]
            self.n_samples_seen_ = 0
            self.mean_ = np.zeros(self.n_features_in_, dtype=float)
            self.var_ = np.zeros(self.n_features_in_, dtype=float)

        if X_chunk.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_chunk.shape[1]}"
            )

        chunk_count = X_chunk.shape[0]
        chunk_mean = np.nanmean(X_chunk, axis=0)
        chunk_var = np.nanvar(X_chunk, axis=0)

        all_nan = np.isnan(chunk_mean)

        if self.n_samples_seen_ == 0:
            chunk_mean = np.where(all_nan, 0.0, chunk_mean)
            chunk_var = np.where(all_nan, 0.0, chunk_var)

            self.mean_ = chunk_mean
            self.var_ = chunk_var
            self.n_samples_seen_ = int(chunk_count)
            return self

        old_count = self.n_samples_seen_
        new_count = old_count + chunk_count

        chunk_mean_safe = np.where(all_nan, self.mean_, chunk_mean)
        chunk_var_safe = np.where(all_nan, self.var_, chunk_var)

        delta = chunk_mean_safe - self.mean_

        new_mean = self.mean_ + delta * chunk_count / new_count
        new_var = (
            old_count * self.var_
            + chunk_count * chunk_var_safe
            + (delta ** 2) * old_count * chunk_count / new_count
        ) / new_count

        self.mean_ = new_mean
        self.var_ = new_var
        self.n_samples_seen_ = int(new_count)

        return self

    def mean(self):
        """
        Return feature-wise running mean.
        """
        if self.n_samples_seen_ == 0:
            raise ValueError("No chunks have been added yet.")

        return self.mean_.copy()

    def variance(self):
        """
        Return feature-wise running population variance.
        """
        if self.n_samples_seen_ == 0:
            raise ValueError("No chunks have been added yet.")

        return self.var_.copy()

    def std(self):
        """
        Return feature-wise running population standard deviation.
        """
        return np.sqrt(self.variance())

    def reset(self):
        """
        Reset all running statistics.
        """
        self.n_samples_seen_ = 0
        self.n_features_in_ = None
        self.mean_ = None
        self.var_ = None

        return self