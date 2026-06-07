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
    StreamingStats,
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
def test_streaming_stats_matches_batch_mean_and_variance():
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    X_all = np.vstack([X1, X2])

    tracker = StreamingStats()
    tracker.update_stats(X1)
    tracker.update_stats(X2)

    assert np.allclose(tracker.mean(), np.mean(X_all, axis=0))
    assert np.allclose(tracker.variance(), np.var(X_all, axis=0))
    assert tracker.n_samples_seen_ == 4


def test_streaming_stats_accepts_1d_input():
    tracker = StreamingStats()
    tracker.update_stats(np.array([1.0, 2.0, 3.0]))

    assert tracker.mean().shape == (1,)
    assert np.isclose(tracker.mean()[0], 2.0)


def test_streaming_stats_reset():
    tracker = StreamingStats()
    tracker.update_stats(np.array([[1.0, 2.0]]))

    tracker.reset()

    assert tracker.n_samples_seen_ == 0

    with pytest.raises(ValueError):
        tracker.mean()


def test_streaming_stats_feature_mismatch_raises():
    tracker = StreamingStats()
    tracker.update_stats(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError):
        tracker.update_stats(np.array([[1.0, 2.0, 3.0]]))


def test_streaming_stats_empty_chunk_raises():
    tracker = StreamingStats()

    with pytest.raises(ValueError):
        tracker.update_stats(np.empty((0, 2)))
def test_validate_array_rejects_non_numpy_input():
    with pytest.raises(TypeError):
        mean([1, 2, 3])


def test_validate_array_rejects_empty_array():
    with pytest.raises(ValueError):
        mean(np.array([]))


def test_histogram_rejects_invalid_bin_count():
    with pytest.raises(ValueError):
        histogram(np.array([1, 2, 3]), n_bins=0)


def test_histogram_rejects_bool_bin_count():
    with pytest.raises(ValueError):
        histogram(np.array([1, 2, 3]), n_bins=True)


def test_quantile_rejects_low_q():
    with pytest.raises(ValueError):
        quantile(np.array([1, 2, 3]), -0.1)


def test_quantile_rejects_high_q():
    with pytest.raises(ValueError):
        quantile(np.array([1, 2, 3]), 1.1)


def test_welford_rejects_non_numeric_value():
    tracker = WelfordStats()

    with pytest.raises(TypeError):
        tracker.update("bad")


def test_welford_single_value_variance_zero():
    tracker = WelfordStats()
    tracker.update(5.0)

    assert tracker.variance() == 0.0


def test_streaming_stats_rejects_3d_chunk():
    tracker = StreamingStats()

    with pytest.raises(ValueError):
        tracker.update_stats(np.ones((2, 2, 2)))


def test_streaming_stats_handles_nan_chunk():
    tracker = StreamingStats()

    tracker.update_stats(np.array([[np.nan, np.nan], [np.nan, np.nan]]))

    assert np.allclose(tracker.mean(), [0.0, 0.0])
    assert np.allclose(tracker.variance(), [0.0, 0.0])