
import numpy as np
import pytest

from numcompute.rank import rank, percentile


def test_average_basics():
    d = np.array([10, 20, 20, 40])
    rst = rank(d, method="average")
    e = np.array([1.0, 2.5, 2.5, 4.0])
    assert np.allclose(rst, e)


def test_dense_basic():
    d = np.array([10, 20, 20, 40])
    rst = rank(d, method="dense")
    e = np.array([1, 2, 2, 3])
    assert np.array_equal(rst, e)


def test_ordinal_basic():
    d = np.array([10, 20, 20, 40])
    rst = rank(d, method="ordinal")
    e = np.array([1, 2, 3, 4])
    assert np.array_equal(rst, e)


def test_all_equal_average():
    d = np.array([5, 5, 5, 5])
    rst = rank(d, method="average")
    e = np.array([2.5, 2.5, 2.5, 2.5])
    assert np.allclose(rst, e)


def test_all_equal_dense():
    d = np.array([5, 5, 5, 5])
    rst = rank(d, method="dense")
    e = np.array([1, 1, 1, 1])
    assert np.array_equal(rst, e)


def test_all_equal_ordinal():
    d = np.array([5, 5, 5, 5])
    r = rank(d, method="ordinal")
    e = np.array([1, 2, 3, 4])
    assert np.array_equal(r, e)


def test_single_value():
    d = np.array([99])
    r = rank(d, method="average")
    e = np.array([1.0])
    assert np.allclose(r, e)


def test_sorted_input():
    d = np.array([1, 2, 3, 4])
    rst = rank(d, method="dense")
    e = np.array([1, 2, 3, 4])
    assert np.array_equal(rst, e)


def test_reverse_sorted_input():
    d = np.array([4, 3, 2, 1])
    rst = rank(d, method="dense")
    e = np.array([4, 3, 2, 1])
    assert np.array_equal(rst, e)


def test_rank_with_negative_values():
    d = np.array([-5, -1, -1, 3])
    rst = rank(d, method="average")
    e = np.array([1.0, 2.5, 2.5, 4.0])
    assert np.allclose(rst, e)


def test_rank_invalid_method():
    with pytest.raises(ValueError):
        rank(np.array([1, 2, 3]), method="bad")


def test_rank_empty_array():
    with pytest.raises(ValueError):
        rank(np.array([]), method="average")


def test_rank_nan_input():
    with pytest.raises(ValueError):
        rank(np.array([1.0, np.nan, 2.0]), method="average")


def test_percentile_50():
    d = np.array([10, 20, 30, 40, 50])
    rst = percentile(d, 50)
    assert rst == 30


def test_percentile_linear():
    d = np.array([10, 20, 30, 40])
    rst = percentile(d, 25, interpolation="linear")
    assert rst == 17.5


def test_percentile_lower():
    d = np.array([10, 20, 30, 40])
    rst = percentile(d, 25, interpolation="lower")
    assert rst == 10


def test_percentile_higher():
    d = np.array([10, 20, 30, 40])
    rst = percentile(d, 25, interpolation="higher")
    assert rst == 20


def test_percentile_midpoint():
    d = np.array([10, 20, 30, 40])
    rst = percentile(d, 25, interpolation="midpoint")
    assert rst == 15.0


def test_percentile_zero():
    d = np.array([10, 20, 30])
    rst = percentile(d, 0)
    assert rst == 10


def test_percentile_hundred():
    d = np.array([10, 20, 30])
    rst = percentile(d, 100)
    assert rst == 30