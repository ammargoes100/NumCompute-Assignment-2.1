"""
Tests for retained sorting and searching utilities.
"""

import numpy as np
import pytest

from numcompute_stream.sort_search import (
    stable_sort,
    multikey_sort,
    topk,
    quickselect,
    binary_search,
)


def test_stable_sort():
    array = np.array([4, 1, 2, 2])

    result = stable_sort(array)
    expected = np.array([1, 2, 2, 4])

    assert np.array_equal(result, expected)


def test_stable_sort_axis_none():
    array = np.array([[5, 1], [2, 0]])

    result = stable_sort(array, axis=None)
    expected = np.array([0, 1, 2, 5])

    assert np.array_equal(result, expected)


def test_stable_sort_reverse_sorted():
    array = np.array([6, 3, 2, 1])

    result = stable_sort(array)
    expected = np.array([1, 2, 3, 6])

    assert np.array_equal(result, expected)


def test_multikey_sort():
    array = np.array([
        [2, 5],
        [1, 8],
        [2, 3],
        [1, 4],
    ])

    result = multikey_sort(array, [0, 1])
    expected = np.array([
        [1, 4],
        [1, 8],
        [2, 3],
        [2, 5],
    ])

    assert np.array_equal(result, expected)


def test_multikey_sort_three_columns():
    array = np.array([
        [1, 2, 3],
        [1, 1, 9],
        [1, 1, 5],
        [2, 0, 1],
    ])

    result = multikey_sort(array, [0, 1, 2])
    expected = np.array([
        [1, 1, 5],
        [1, 1, 9],
        [1, 2, 3],
        [2, 0, 1],
    ])

    assert np.array_equal(result, expected)


def test_multikey_sort_empty_keys():
    array = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        multikey_sort(array, [])


def test_multikey_sort_invalid_dimension():
    with pytest.raises(ValueError):
        multikey_sort(np.array([1, 2, 3]), [0])


def test_multikey_sort_key_out_of_bounds():
    array = np.array([[1, 2], [3, 4]])

    with pytest.raises(IndexError):
        multikey_sort(array, [2])


def test_topk_largest():
    array = np.array([4, 1, 9, 7, 3])

    result = topk(array, 2, largest=True, return_indices=False)
    expected = np.array([9, 7])

    assert np.array_equal(result, expected)


def test_topk_smallest():
    array = np.array([4, 1, 9, 7, 3])

    result = topk(array, 2, largest=False, return_indices=False)
    expected = np.array([1, 3])

    assert np.array_equal(result, expected)


def test_topk_k_equals_length():
    array = np.array([4, 1, 9])

    result = topk(array, 3, largest=True, return_indices=False)
    expected = np.array([9, 4, 1])

    assert np.array_equal(result, expected)


def test_topk_k_one():
    array = np.array([4, 1, 9, 6, 3])

    result = topk(array, 1, largest=True, return_indices=False)
    expected = np.array([9])

    assert np.array_equal(result, expected)


def test_topk_return_indices():
    array = np.array([4, 1, 9, 7, 3])

    values, indices = topk(array, 2, largest=True, return_indices=True)

    assert np.array_equal(values, array[indices])
    assert set(values.tolist()) == {9, 7}


def test_topk_invalid_k_zero():
    with pytest.raises(ValueError):
        topk(np.array([1, 2, 3]), 0)


def test_topk_invalid_k_too_large():
    with pytest.raises(ValueError):
        topk(np.array([1, 2, 3]), 4)


def test_topk_empty_array():
    with pytest.raises(ValueError):
        topk(np.array([]), 1)


def test_topk_non_contiguous_array():
    array = np.arange(10)[::2]

    result = topk(array, 2, largest=True, return_indices=False)
    expected = np.array([8, 6])

    assert np.array_equal(result, expected)


def test_quickselect_minimum():
    array = np.array([7, 2, 9, 1, 5])

    result = quickselect(array, 0)

    assert result == 1


def test_quickselect_middle():
    array = np.array([7, 2, 9, 1, 5])

    result = quickselect(array, 2)

    assert result == 5


def test_binary_search_found():
    index, exists = binary_search(np.array([1, 3, 5, 7]), 5)

    assert index == 2
    assert exists is True