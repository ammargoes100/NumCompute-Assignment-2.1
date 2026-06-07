"""
Tests for retained descriptive statistics utilities.
"""

import numpy as np
import pytest

from numcompute_stream.stats import (
    mean,
    median,
    std,
    minimum,
    maximum,
    histogram,
    quantile,
    WelfordStats,
)


def test_mean():
    assert np.isclose(mean(np.array([1, 2, 3, 4])), 2.5)

    arr2d = np.array([[1, 2], [3, 4]])
    assert np.array_equal(mean(arr2d, axis=0), np.array([2.0, 3.0]))
    assert np.array_equal(mean(arr2d, axis=1), np.array([1.5, 3.5]))

    arr3d = np.arange(1, 9).reshape(2, 2, 2)
    assert np.isclose(mean(arr3d), 4.5)


def test_median():
    arr = np.array([1, np.nan, 3, 4, 5])
    assert np.isclose(median(arr), 3.5)

    assert np.isnan(median(np.array([np.nan, np.nan])))

    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(median(arr2d, axis=0), np.array([2.5, 3.5, 4.5]))


def test_std():
    arr = np.array([2, 4, 4, 4, 5, 5, 7, 9])
    assert np.isclose(std(arr), 2.0)

    assert std(np.array([42])) == 0.0

    arr2d = np.array([[1, 2], [3, 4]])
    assert np.array_equal(std(arr2d, axis=0), np.array([1.0, 1.0]))


def test_minimum_and_maximum():
    arr = np.array([3, 1, 4, np.nan, 2])

    assert np.isclose(minimum(arr), 1.0)
    assert np.isclose(maximum(arr), 4.0)

    arr3d = np.array([
        [[5, 1], [3, 7]],
        [[9, 2], [0, 4]],
    ])

    assert minimum(arr3d) == 0
    assert maximum(arr3d) == 9


def test_histogram():
    arr = np.array([1, 1, 2, 2, 2, 3])

    counts, edges = histogram(arr, n_bins=3)

    assert counts.sum() == len(arr)
    assert len(edges) == 4

    all_nan = np.array([np.nan, np.nan])
    counts_nan, edges_nan = histogram(all_nan)

    assert counts_nan.sum() == 0
    assert np.isnan(edges_nan).all()


def test_quantile():
    arr = np.array([1, 2, 3, 4, 5])

    assert np.isclose(quantile(arr, 0.0), 1)
    assert np.isclose(quantile(arr, 1.0), 5)
    assert np.isclose(quantile(arr, 0.5), 3)

    qs = quantile(arr, [0.25, 0.5, 0.75])
    assert qs[0] <= qs[1] <= qs[2]

    arr2d = np.array([[1, 2], [3, 4], [5, 6]])
    q_col = quantile(arr2d, 0.5, axis=0)

    assert np.array_equal(q_col, np.array([3.0, 4.0]))


def test_welford_stats():
    tracker = WelfordStats()
    data = [1, 2, 3, 4, 5]

    for value in data:
        tracker.update(value)

    assert np.isclose(tracker.mean(), 3.0)
    assert np.isclose(tracker.std(), np.std(data))

    tracker_nan = WelfordStats()
    tracker_nan.update(1).update(np.nan).update(2)

    assert np.isclose(tracker_nan.mean(), 1.5)

    tracker.reset()

    assert tracker.n_samples == 0

    with pytest.raises(ValueError):
        tracker.mean()