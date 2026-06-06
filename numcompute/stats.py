
# numcompute/stats.py

"""
stats.py - Descriptive statistics for NumCompute.

Provides:
- Basic descriptive statistics: mean, median, std, min, max
- Histogram
- Quantiles with NaN handling
- Axis-wise stats with clear dimension/shape behaviour
- Streaming statistics using Welford's algorithm
"""

import numpy as np


def _validate_array(data):
    """Raise TypeError if data is not a np.ndarray, ValueError if empty."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a NumPy array, got {type(data).__name__}.")
    if data.size == 0:
        raise ValueError("data must not be empty.")


def mean(data, axis=None):
    """
    Compute mean, ignoring NaN values.

    Parameters:
    data : np.ndarray - input array, any shape
    axis : int or None - 0=column-wise, 1=row-wise, None=entire array

    Returns:
    np.ndarray or float

    Raises:
    TypeError  - if data is not np.ndarray
    ValueError - if data is empty

    Time complexity : O(n) | Space complexity : O(1)
    """
    _validate_array(data)
    with np.errstate(all='ignore'):
        return np.nanmean(data, axis=axis)


def median(data, axis=None):
    """
    Compute median, ignoring NaN values.

    Parameters:
    data : np.ndarray - input array, any shape
    axis : int or None - 0=column-wise, 1=row-wise, None=entire array

    Returns:
    np.ndarray or float

    Raises:
    TypeError  - if data is not np.ndarray
    ValueError - if data is empty

    Time complexity : O(n log n) | Space complexity : O(n)
    """
    _validate_array(data)
    #check if data consisit only of NaNs
    if np.all(np.isnan(data)):
        return np.nan
    with np.errstate(all='ignore'):
        return np.nanmedian(data, axis=axis)


def std(data, axis=None):
    """
    Compute standard deviation, ignoring NaN values.

    >>Parameters:
    data : np.ndarray - input array, any shape
    axis : int or None - 0=column-wise, 1=row-wise, None=entire array

    Returns:
        np.ndarray or float

    >>Raises:
    TypeError  - if data is not np.ndarray
    ValueError - if data is empty

    Time complexity : O(n) | Space complexity : O(1)
    """
    _validate_array(data)
    with np.errstate(all='ignore'):
        return np.nanstd(data, axis=axis)


def minimum(data, axis=None):
    """
    Compute minimum, ignoring NaN values.

    Parameters:
    data : np.ndarray - input array, any shape
    axis : int or None - 0=column-wise, 1=row-wise, None=entire array

    Returns:
    np.ndarray or float

    Raises:
    TypeError  - if data is not np.ndarray
    ValueError - if data is empty

    Time complexity : O(n) | Space complexity : O(1)
    """
    _validate_array(data)
    with np.errstate(all='ignore'):
        return np.nanmin(data, axis=axis)


def maximum(data, axis=None):
    """
    Compute maximum, ignoring NaN values.

    Parameters:
    data : np.ndarray - input array, any shape
    axis : int or None - 0=column-wise, 1=row-wise, None=entire array

    Returns:
    np.ndarray or float

    Raises:
    TypeError  - if data is not np.ndarray
    ValueError - if data is empty

    Time complexity : O(n) | Space complexity : O(1)
    """
    _validate_array(data)
    with np.errstate(all='ignore'):
        return np.nanmax(data, axis=axis)


def histogram(data, n_bins=10):
    """
    Compute histogram, ignoring NaN values.

    Parameters:
    data   : np.ndarray - input array, any shape (flattened internally)
    n_bins : int        - number of bins (default 10)

    Returns:
    counts    : np.ndarray - shape (n_bins,) values per bin
    bin_edges : np.ndarray - shape (n_bins+1,) bin boundaries

    Raises:
    TypeError  - if data is not np.ndarray
    ValueError - if data is empty or n_bins < 1

    Time complexity : O(n log n) | Space complexity : O(n_bins)
    """
    _validate_array(data)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}.")

    valid_data = data.flatten()
    valid_data = valid_data[~np.isnan(valid_data)]

    # Handle all-NaN case
    if len(valid_data) == 0:
        return np.zeros(n_bins, dtype=int), np.array([np.nan] * (n_bins + 1))

    return np.histogram(valid_data, bins=n_bins)


def quantile(data, q, axis=None):
    """
    Compute quantiles, ignoring NaN values.

    Parameters:
    data : np.ndarray    - input array, any shape
    q    : float or list - quantile(s) between 0 and 1
    axis : int or None   - 0=column-wise, 1=row-wise, None=entire array

    Returns:
    np.ndarray or float

    Raises:
    TypeError  - if data is not np.ndarray
    ValueError - if data is empty or q not between 0 and 1

    Time complexity : O(n log n) | Space complexity : O(n)
    """
    _validate_array(data)
    q_array = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if np.any(q_array < 0) or np.any(q_array > 1):
        raise ValueError(f"q values must be between 0 and 1, got {q}.")
    with np.errstate(all='ignore'):
        return np.nanquantile(data, q, axis=axis)


class WelfordStats:
    """
    Incremental mean and variance using Welford's algorithm.

    Use when data arrives one value at a time and storing
    all values in memory is not practical.
    """

    def __init__(self):
        """Initialise with zero samples."""
        self.n_samples = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, new_value):
        """
        Update running statistics with a new value. NaNs are ignored.

        Parameters:
        new_value : float - next value in the stream

        Returns:
        self - for method chaining

        Time complexity : O(1) | Space complexity : O(1)
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

        Raises:
        ValueError - if no values have been added yet

        Time complexity : O(1) | Space complexity : O(1)
        """
        if self.n_samples == 0:
            raise ValueError("No values added yet. Call update() first.")
        return self._mean

    def variance(self):
        """
        Return current population variance.

        Raises:
        ValueError - if no values have been added yet

        Time complexity : O(1) | Space complexity : O(1)
        """
        if self.n_samples == 0:
            raise ValueError("No values added yet. Call update() first.")
        if self.n_samples == 1:
            return 0.0
        return self._M2 / self.n_samples

    def std(self):
        """
        Return current population standard deviation.

        Time complexity : O(1) | Space complexity : O(1)
        """
        return np.sqrt(self.variance())

    def reset(self):
        """
        Reset tracker to initial state.

        Returns
        -------
        self - for method chaining

        Time complexity : O(1) | Space complexity : O(1)
        """
        self.n_samples = 0
        self._mean = 0.0
        self._M2 = 0.0
        return self