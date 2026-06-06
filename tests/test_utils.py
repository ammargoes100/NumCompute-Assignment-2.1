"""
Tests for utils.py
"""
import numpy as np
import pytest
from numcompute.utils import logsumexp, softmax, euclidean_distance, pairwise_euclidean, manhattan_distance, tanh, topk_indices, cosine_similarity,batch_iter, relu


def test_logsumexp_matches_naive():
    # on small values naive and stable should give same answer
    x = np.array([1.0, 2.0, 3.0])
    expected = np.log(np.sum(np.exp(x)))
    result = logsumexp(x)
    assert np.isclose(result, expected)


def test_logsumexp_stable_on_large_values():
    # naive exp(1000) would overflow to inf, stable version should not
    x = np.array([1000.0, 1001.0, 1002.0])
    result = logsumexp(x)
    assert np.isfinite(result)


def test_softmax_sums_to_one():
    # softmax output must always sum to 1
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = softmax(x, axis=0)
    assert np.isclose(np.sum(result), 1.0)


def test_euclidean_distance_known_result():
    # distance between [0,0] and [3,4] is 5
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert np.isclose(euclidean_distance(a, b), 5.0)


def test_pairwise_euclidean_output_shape():
    # output should be (n, m) for inputs (n, d) and (m, d)
    X = np.random.rand(5, 3)
    Y = np.random.rand(4, 3)
    result = pairwise_euclidean(X, Y)
    assert result.shape == (5, 4)

def test_relu_negatives_zeroed():
    x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    result = relu(x)
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 2.0, 5.0])


def test_tanh_range():
    x = np.array([-100.0, 0.0, 100.0])
    result = tanh(x)
    assert np.all(result >= -1) and np.all(result <= 1)
    assert np.isclose(result[1], 0.0)


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


def test_pairwise_euclidean_values():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[3.0, 4.0]])
    result = pairwise_euclidean(X, Y)
    assert np.isclose(result[0, 0], 5.0)


def test_topk_indices_largest():
    values = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
    idx = topk_indices(values, k=2, largest=True)
    assert set(idx) == {3, 1}


def test_topk_indices_smallest():
    values = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
    idx = topk_indices(values, k=2, largest=False)
    assert set(idx) == {0, 4}


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
    assert all(len(xb) == len(yb) for xb, yb in batches)