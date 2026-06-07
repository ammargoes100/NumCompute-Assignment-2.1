"""
Sorting and searching utilities for NumCompute-Stream.

This module keeps the sorting, top-k, quickselect, and binary search helpers
from the original NumCompute package. They are retained because they are useful
general numerical utilities for array manipulation and analysis.
"""

import numpy as np


def stable_sort(array, axis=-1):
    """
    Sort an array using NumPy's stable sorting algorithm.

    Parameters
    ----------
    array : array-like
        Input array.
    axis : int or None, default=-1
        Axis along which to sort. If None, the array is flattened first.

    Returns
    -------
    np.ndarray
        Sorted array.
    """
    array = np.asarray(array)

    if axis is None:
        return np.sort(array.reshape(-1), kind="stable")

    return np.sort(array, axis=axis, kind="stable")


def multikey_sort(array, keys):
    """
    Sort a 2D array by multiple column keys.

    Parameters
    ----------
    array : array-like of shape (n_samples, n_features)
        Input 2D array.
    keys : list of int
        Column indices used for sorting priority.

    Returns
    -------
    np.ndarray
        Sorted array.
    """
    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError("multikey_sort expects a 2D array")

    if keys is None or len(keys) == 0:
        raise ValueError("at least one column index is required")

    n_cols = array.shape[1]

    for col in keys:
        if not isinstance(col, (int, np.integer)):
            raise ValueError("all keys must be integer column indices")

        if col < 0 or col >= n_cols:
            raise IndexError(
                f"Column index {col} is out of bounds for array with {n_cols} columns."
            )

    order = np.lexsort([array[:, col] for col in reversed(keys)])
    return array[order]


def topk(values, k, largest=True, return_indices=True, sorted=True):
    """
    Return the top-k largest or smallest values from a 1D array.

    Parameters
    ----------
    values : array-like of shape (n,)
        Input values.
    k : int
        Number of values to return.
    largest : bool, default=True
        If True, return largest values. Otherwise return smallest values.
    return_indices : bool, default=True
        If True, return both values and original indices.
    sorted : bool, default=True
        If True, sort the selected values.

    Returns
    -------
    np.ndarray or tuple
        Selected values, and optionally their original indices.
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError("topk expects a 1D array")

    if values.size == 0:
        raise ValueError("topk cannot operate on an empty array")

    if not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")

    if k < 1 or k > values.size:
        raise ValueError(f"k must satisfy 1 <= k <= {values.size}")

    if k == values.size:
        indices = np.arange(values.size)
    else:
        if largest:
            indices = np.argpartition(values, -k)[-k:]
        else:
            indices = np.argpartition(values, k - 1)[:k]

    selected_values = values[indices]

    if sorted:
        if largest:
            order = np.argsort(-selected_values, kind="stable")
        else:
            order = np.argsort(selected_values, kind="stable")

        indices = indices[order]
        selected_values = selected_values[order]

    if return_indices:
        return selected_values, indices

    return selected_values


def quickselect(array, k):
    """
    Return the k-th smallest element from a 1D array.

    Parameters
    ----------
    array : array-like of shape (n,)
        Input values.
    k : int
        Zero-based index of the desired smallest value.

    Returns
    -------
    scalar
        The k-th smallest value.
    """
    array = np.asarray(array)

    if array.ndim != 1:
        raise ValueError("quickselect expects a 1D array")

    if array.size == 0:
        raise ValueError("quickselect cannot operate on an empty array")

    if not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")

    if k < 0 or k >= array.size:
        raise ValueError(f"k must satisfy 0 <= k < {array.size}")

    work = array.copy()
    left = 0
    right = work.size - 1

    while True:
        if left == right:
            return work[left]

        pivot_index = (left + right) // 2
        pivot_value = work[pivot_index]

        work[pivot_index], work[right] = work[right], work[pivot_index]

        store_index = left

        for i in range(left, right):
            if work[i] < pivot_value:
                work[store_index], work[i] = work[i], work[store_index]
                store_index += 1

        work[right], work[store_index] = work[store_index], work[right]

        if k == store_index:
            return work[k]

        if k < store_index:
            right = store_index - 1
        else:
            left = store_index + 1


def binary_search(sorted_array, target):
    """
    Search for a value in a sorted 1D array.

    Parameters
    ----------
    sorted_array : array-like of shape (n,)
        Sorted input array.
    target : scalar
        Value to search for.

    Returns
    -------
    tuple
        (index, exists), where index is the insertion/found position and exists
        indicates whether the target was found.
    """
    sorted_array = np.asarray(sorted_array)

    if sorted_array.ndim != 1:
        raise ValueError("binary_search expects a 1D array")

    index = int(np.searchsorted(sorted_array, target, side="left"))
    exists = index < sorted_array.size and sorted_array[index] == target

    return index, bool(exists)