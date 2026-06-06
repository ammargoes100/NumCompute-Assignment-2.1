import pytest
import numpy as np
from numcompute.stats import (
    mean, median, std, minimum, maximum,
    histogram, quantile, WelfordStats
)

# Test 1: mean() with 1D, 2D, 3D (including axis)

def test_mean():
    # 1D
    assert np.isclose(mean(np.array([1,2,3,4])), 2.5)
    # 2D axis=0 (column-wise)
    arr2d = np.array([[1,2],[3,4]])
    assert np.array_equal(mean(arr2d, axis=0), np.array([2.,3.]))
    # 2D axis=1 (row-wise)
    assert np.array_equal(mean(arr2d, axis=1), np.array([1.5, 3.5]))
    # 3D (flattened, axis=None)
    arr3d = np.arange(1, 9).reshape(2,2,2)
    assert np.isclose(mean(arr3d), 4.5)


# Test 2: median() with NaN handling

def test_median():
    arr = np.array([1, np.nan, 3, 4, 5])
    assert np.isclose(median(arr), 3.5)  # median of [1,3,4,5] -> (3+4)/2
    # all NaN -> returns NaN
    assert np.isnan(median(np.array([np.nan, np.nan])))
    # 2D axis
    arr2d = np.array([[1,2,3],[4,5,6]])
    assert np.array_equal(median(arr2d, axis=0), np.array([2.5, 3.5, 4.5]))


# Test 3: std() with single value and axis

def test_std():
    arr = np.array([2,4,4,4,5,5,7,9])
    assert np.isclose(std(arr), 2.0)
    assert std(np.array([42])) == 0.0
    # 2D axis
    arr2d = np.array([[1,2],[3,4]])
    assert np.array_equal(std(arr2d, axis=0), np.array([1.,1.]))


# Test 4: minimum() and maximum() with NaNs

def test_min_max():
    arr = np.array([3,1,4,np.nan,2])
    assert np.isclose(minimum(arr), 1.0)
    assert np.isclose(maximum(arr), 4.0)
    # 3D axis
    arr3d = np.array([[[5,1],[3,7]],[[9,2],[0,4]]])
    assert minimum(arr3d) == 0
    assert maximum(arr3d) == 9


# Test 5: histogram() with normal and all-NaN

def test_histogram():
    arr = np.array([1,1,2,2,2,3])
    counts, edges = histogram(arr, n_bins=3)
    assert counts.sum() == len(arr)
    assert len(edges) == 4
    # all-NaN should not crash and return zeros + NaN edges
    all_nan = np.array([np.nan, np.nan])
    c, e = histogram(all_nan)
    assert c.sum() == 0
    assert np.isnan(e).all()


# Test 6: quantile() with 2D axis and multiple quantiles

def test_quantile():
    arr = np.array([1,2,3,4,5])
    assert np.isclose(quantile(arr, 0.0), 1)
    assert np.isclose(quantile(arr, 1.0), 5)
    assert np.isclose(quantile(arr, 0.5), 3)
    # multiple quantiles
    qs = quantile(arr, [0.25, 0.5, 0.75])
    assert qs[0] <= qs[1] <= qs[2]
    # 2D axis
    arr2d = np.array([[1,2],[3,4],[5,6]])
    q_col = quantile(arr2d, 0.5, axis=0)
    assert np.array_equal(q_col, np.array([3.,4.]))


# Test 7: WelfordStats streaming (with NaN skip and reset)

def test_welford():
    tracker = WelfordStats()
    data = [1,2,3,4,5]
    for x in data:
        tracker.update(x)
    assert np.isclose(tracker.mean(), 3.0)
    assert np.isclose(tracker.std(), np.std(data))
    # Test NaN skipping
    tracker_nan = WelfordStats()
    tracker_nan.update(1).update(np.nan).update(2)
    assert np.isclose(tracker_nan.mean(), 1.5)
    # Test reset
    tracker.reset()
    assert tracker.n_samples == 0
    with pytest.raises(ValueError):
        tracker.mean()