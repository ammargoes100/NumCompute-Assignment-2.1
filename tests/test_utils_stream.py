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
    chunk_iter,
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
def test_logsumexp_rejects_empty_input():
    with pytest.raises(ValueError):
        logsumexp(np.array([]))


def test_softmax_rejects_empty_input():
    with pytest.raises(ValueError):
        softmax(np.array([]))


def test_distance_shape_mismatch_raises():
    with pytest.raises(ValueError):
        euclidean_distance(np.array([1.0, 2.0]), np.array([1.0]))


def test_manhattan_shape_mismatch_raises():
    with pytest.raises(ValueError):
        manhattan_distance(np.array([1.0, 2.0]), np.array([1.0]))


def test_cosine_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.array([1.0, 2.0]), np.array([1.0]))


def test_pairwise_euclidean_rejects_non_2d_input():
    with pytest.raises(ValueError):
        pairwise_euclidean(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))


def test_pairwise_euclidean_feature_mismatch_raises():
    X = np.ones((2, 3))
    Y = np.ones((4, 2))

    with pytest.raises(ValueError):
        pairwise_euclidean(X, Y)


def test_topk_indices_rejects_empty_values():
    with pytest.raises(ValueError):
        topk_indices(np.array([]), k=1)


def test_topk_indices_rejects_2d_values():
    with pytest.raises(ValueError):
        topk_indices(np.array([[1, 2], [3, 4]]), k=1)


def test_topk_indices_rejects_non_integer_k():
    with pytest.raises(ValueError):
        topk_indices(np.array([1, 2, 3]), k=1.5)


def test_topk_indices_rejects_bool_k():
    with pytest.raises(ValueError):
        topk_indices(np.array([1, 2, 3]), k=True)


def test_batch_iter_rejects_empty_X():
    with pytest.raises(ValueError):
        list(batch_iter(np.empty((0, 2)), batch_size=2))


def test_batch_iter_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        list(batch_iter(np.arange(10), batch_size=0))


def test_batch_iter_rejects_non_integer_batch_size():
    with pytest.raises(ValueError):
        list(batch_iter(np.arange(10), batch_size=2.5))


def test_batch_iter_rejects_y_length_mismatch():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(4)

    with pytest.raises(ValueError):
        list(batch_iter(X, batch_size=2, y=y))


def test_batch_iter_shuffle_is_reproducible():
    X = np.arange(20).reshape(10, 2)

    first = list(batch_iter(X, batch_size=3, shuffle=True, seed=42))
    second = list(batch_iter(X, batch_size=3, shuffle=True, seed=42))

    first_flat = np.vstack(first)
    second_flat = np.vstack(second)

    assert np.array_equal(first_flat, second_flat)


def test_chunk_iter_matches_batch_iter():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)

    chunks = list(chunk_iter(X, chunk_size=2, y=y))

    assert len(chunks) == 3
    assert chunks[0][0].shape == (2, 2)
    assert chunks[0][1].shape == (2,)