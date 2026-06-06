import numpy as np
import pytest

from numcompute.sort_search import (
    stable_sort,
    multikey_sort,
    topk,
    quickselect,
    binary_search,
)


def test_stable_sort():
    ar = np.array([4, 1, 2, 2])
    rst = stable_sort(ar)
    e = np.array([1, 2, 2, 4])
    assert np.array_equal(rst, e)


def test_stable_sort_axis_none():
    ar = np.array([[5, 1], [2, 0]])
    rst = stable_sort(ar, axis=None)
    e = np.array([0, 1, 2, 5])
    assert np.array_equal(rst, e)


def test_stable_sort_reverse_sorted():
    ar = np.array([6, 3, 2, 1])
    rst = stable_sort(ar)
    e = np.array([1, 2, 3, 6])
    assert np.array_equal(rst, e)


def test_multikey_sort():
    x = np.array([
        [2, 5],
        [1, 8],
        [2, 3],
        [1, 4],
    ])
    rst = multikey_sort(x, [0, 1])
    e = np.array([
        [1, 4],
        [1, 8],
        [2, 3],
        [2, 5],
    ])
    assert np.array_equal(rst, e)


def test_multikey_sort_three_columns():
    x = np.array([
        [1, 2, 3],
        [1, 1, 9],
        [1, 1, 5],
        [2, 0, 1],
    ])
    rst = multikey_sort(x, [0, 1, 2])
    e = np.array([
        [1, 1, 5],
        [1, 1, 9],
        [1, 2, 3],
        [2, 0, 1],
    ])
    assert np.array_equal(rst, e)


def test_multikey_sort_emptykeys():
    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        multikey_sort(x, [])


def test_multikey_sort_invalid_dimension():
    with pytest.raises(ValueError):
        multikey_sort(np.array([1, 2, 3]), [0])


def test_multikey_sort_key_outofbounds():
    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(IndexError):
        multikey_sort(x, [2])


def test_topk_largest():
    ar = np.array([4, 1, 9, 7, 3])
    rst = topk(ar, 2, largest=True, return_indices=False)
    e = np.array([9, 7])
    assert np.array_equal(rst, e)


def test_topk_smallest():
    ar = np.array([4, 1, 9, 7, 3])
    rst = topk(ar, 2, largest=False, return_indices=False)
    e = np.array([1, 3])
    assert np.array_equal(rst, e)


def test_topk_k_equals_length():
    ar = np.array([4, 1, 9])
    rst = topk(ar, 3, largest=True, return_indices=False)
    e = np.array([9, 4, 1])
    assert np.array_equal(rst, e)


def test_topk_k_one():
    ar = np.array([4, 1, 9, 6, 3])
    rst = topk(ar, 1, largest=True, return_indices=False)
    e = np.array([9])
    assert np.array_equal(rst, e)


def test_topk_return_indices():
    ar = np.array([4, 1, 9, 7, 3])
    values, indices = topk(ar, 2, largest=True, return_indices=True)
    assert np.array_equal(values, ar[indices])
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
    ar = np.arange(10)[::2]  # [0, 2, 5, 6, 8]
    rst = topk(ar, 2, largest=True, return_indices=False)
    e = np.array([8, 6])
    assert np.array_equal(rst, e)


def test_quickselect_minimum():
    ar = np.array([7, 2, 9, 1, 5])
    rst = quickselect(ar, 0)
    assert rst == 1


def test_quickselect_middle():
    ar = np.array([7, 2, 9, 1, 5])
    rst = quickselect(ar, 2)
    assert rst == 5


def test_binary_search_found():
    idx, exists = binary_search(np.array([1, 3, 5, 7]), 5)
    assert idx == 2
    assert exists is True
