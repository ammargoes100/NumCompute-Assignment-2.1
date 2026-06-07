"""
Utility functions for NumCompute-Stream.

This module keeps the general numerical helpers from the original NumCompute
package. The stable activation functions, distance metrics, top-k helper, and
batch iterator are retained because they are useful across preprocessing,
models, benchmarking, and demos.
"""

import numpy as np


def logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    """
    x = np.asarray(x)
    max_value = np.max(x, axis=axis, keepdims=True)
    result = max_value + np.log(
        np.sum(np.exp(x - max_value), axis=axis, keepdims=True)
    )

    if axis is not None:
        result = np.squeeze(result, axis=axis)

    return result


def softmax(x, axis=None):
    """
    Compute softmax in a numerically stable way.
    """
    x = np.asarray(x)
    max_value = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_value)

    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x):
    """
    Compute the sigmoid activation element-wise.
    """
    x = np.asarray(x, dtype=float)

    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def relu(x):
    """
    Compute the ReLU activation element-wise.
    """
    return np.maximum(0, np.asarray(x, dtype=float))


def tanh(x):
    """
    Compute the tanh activation element-wise.
    """
    return np.tanh(np.asarray(x, dtype=float))


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(a, b):
    """
    Compute Manhattan distance between two vectors.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    return float(np.sum(np.abs(a - b)))


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero-norm vectors.")

    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_euclidean(X, Y):
    """
    Compute pairwise Euclidean distances between two sets of vectors.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T

    distances_squared = X_norm + Y_norm - 2 * np.dot(X, Y.T)

    return np.sqrt(np.maximum(distances_squared, 0))


def topk_indices(values, k, largest=True):
    """
    Return indices of the top-k values without full sorting.
    """
    values = np.asarray(values)
    n = len(values)

    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and {n}, got {k}.")

    if largest:
        return np.argpartition(values, n - k)[n - k:]

    return np.argpartition(values, k)[:k]


def batch_iter(X, batch_size, y=None, shuffle=False, seed=None):
    """
    Yield mini-batches from X and optionally y.
    """
    X = np.asarray(X)
    n = X.shape[0]

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size]

        if y is not None:
            yield X[batch_indices], np.asarray(y)[batch_indices]
        else:
            yield X[batch_indices]