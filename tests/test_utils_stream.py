"""
Tests for retained utility functions.
"""

import numpy as np
import pytest

from numcompute_stream.utils import (
    logsumexp,
    softmax,
    sigmoid,
    relu,
    tanh,
    euclidean_distance,
    manhattan_distance,
    cosine_similarity,
    pairwise_euclidean,
    topk_indices,
    batch_iter,
)


def test_logsumexp_matches_naive():
    x = np.array([1.0, 2.0, 3.0])

    expected = np.log(np.sum(np.exp(x)))
    result = logsumexp(x)

    assert np.isclose(result, expected)


def test_logsumexp_stable_on_large_values():
    x = np.array([1000.0, 1001.0, 1002.0])

    result = logsumexp(x)

    assert np.isfinite(result)


def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])

    result = softmax(x, axis=0)

    assert np.isclose(np.sum(result), 1.0)


def test_sigmoid_range():
    x = np.array([-100.0, 0.0, 100.0])

    result = sigmoid(x)

    assert np.all(result >= 0)
    assert np.all(result <= 1)
    assert np.isclose(result[1], 0.5)


def test_relu_negatives_zeroed():
    x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])

    result = relu(x)

    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 2.0, 5.0])


def test_tanh_range():
    x = np.array([-100.0, 0.0, 100.0])

    result = tanh(x)

    assert np.all(result >= -1)
    assert np.all(result <= 1)
    assert np.isclose(result[1], 0.0)


def test_euclidean_distance_known_result():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])

    assert np.isclose(euclidean_distance(a, b), 5.0)


def test_manhattan_distance_known_result():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])

    assert np.isclose(manhattan_distance(a, b), 7.0)


def test_cosine_similarity_identical_vectors():
    a = np.array([1.0, 2.0, 3.0])

    assert np.isclose(cosine_similarity(a, a), 1.0)


def test_cosine_similarity_zero_norm_raises():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_pairwise_euclidean_output_shape():
    X = np.random.default_rng(42).random((5, 3))
    Y = np.random.default_rng(123).random((4, 3))

    result = pairwise_euclidean(X, Y)

    assert result.shape == (5, 4)


def test_pairwise_euclidean_values():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[3.0, 4.0]])

    result = pairwise_euclidean(X, Y)

    assert np.isclose(result[0, 0], 5.0)


def test_topk_indices_largest():
    values = np.array([1.0, 5.0, 3.0, 9.0, 2.0])

    indices = topk_indices(values, k=2, largest=True)

    assert set(indices.tolist()) == {3, 1}


def test_topk_indices_smallest():
    values = np.array([1.0, 5.0, 3.0, 9.0, 2.0])

    indices = topk_indices(values, k=2, largest=False)

    assert set(indices.tolist()) == {0, 4}


def test_topk_indices_bad_k_raises():
    with pytest.raises(ValueError):
        topk_indices(np.array([1.0, 2.0]), k=5)


def test_batch_iter_without_y():
    X = np.arange(10).reshape(5, 2)

    batches = list(batch_iter(X, batch_size=2))

    assert len(batches) == 3
    assert batches[0].shape == (2, 2)


def test_batch_iter_with_y():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)

    batches = list(batch_iter(X, batch_size=2, y=y))

    assert all(len(X_batch) == len(y_batch) for X_batch, y_batch in batches)