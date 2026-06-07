"""
Tests for retained ranking and percentile utilities.
"""

import numpy as np
import pytest

from numcompute_stream.rank import rank, percentile


def test_average_basic():
    values = np.array([10, 20, 20, 40])

    result = rank(values, method="average")
    expected = np.array([1.0, 2.5, 2.5, 4.0])

    assert np.allclose(result, expected)


def test_dense_basic():
    values = np.array([10, 20, 20, 40])

    result = rank(values, method="dense")
    expected = np.array([1, 2, 2, 3])

    assert np.array_equal(result, expected)


def test_ordinal_basic():
    values = np.array([10, 20, 20, 40])

    result = rank(values, method="ordinal")
    expected = np.array([1, 2, 3, 4])

    assert np.array_equal(result, expected)


def test_all_equal_average():
    values = np.array([5, 5, 5, 5])

    result = rank(values, method="average")
    expected = np.array([2.5, 2.5, 2.5, 2.5])

    assert np.allclose(result, expected)


def test_all_equal_dense():
    values = np.array([5, 5, 5, 5])

    result = rank(values, method="dense")
    expected = np.array([1, 1, 1, 1])

    assert np.array_equal(result, expected)


def test_all_equal_ordinal():
    values = np.array([5, 5, 5, 5])

    result = rank(values, method="ordinal")
    expected = np.array([1, 2, 3, 4])

    assert np.array_equal(result, expected)


def test_single_value_rank():
    values = np.array([99])

    result = rank(values, method="average")
    expected = np.array([1.0])

    assert np.allclose(result, expected)


def test_sorted_input_dense():
    values = np.array([1, 2, 3, 4])

    result = rank(values, method="dense")
    expected = np.array([1, 2, 3, 4])

    assert np.array_equal(result, expected)


def test_reverse_sorted_input_dense():
    values = np.array([4, 3, 2, 1])

    result = rank(values, method="dense")
    expected = np.array([4, 3, 2, 1])

    assert np.array_equal(result, expected)


def test_rank_with_negative_values():
    values = np.array([-5, -1, -1, 3])

    result = rank(values, method="average")
    expected = np.array([1.0, 2.5, 2.5, 4.0])

    assert np.allclose(result, expected)


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
    values = np.array([10, 20, 30, 40, 50])

    result = percentile(values, 50)

    assert result == 30


def test_percentile_linear():
    values = np.array([10, 20, 30, 40])

    result = percentile(values, 25, interpolation="linear")

    assert result == 17.5


def test_percentile_lower():
    values = np.array([10, 20, 30, 40])

    result = percentile(values, 25, interpolation="lower")

    assert result == 10


def test_percentile_higher():
    values = np.array([10, 20, 30, 40])

    result = percentile(values, 25, interpolation="higher")

    assert result == 20


def test_percentile_midpoint():
    values = np.array([10, 20, 30, 40])

    result = percentile(values, 25, interpolation="midpoint")

    assert result == 15.0


def test_percentile_zero():
    values = np.array([10, 20, 30])

    result = percentile(values, 0)

    assert result == 10


def test_percentile_hundred():
    values = np.array([10, 20, 30])

    result = percentile(values, 100)

    assert result == 30