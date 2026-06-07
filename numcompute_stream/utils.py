"""
Utility functions for NumCompute-Stream.

This module keeps the general numerical helpers from the original NumCompute
package. The stable activation functions, distance metrics, top-k helper, and
batch iterator are retained because they are useful across preprocessing,
models, benchmarking, and demos.

Small validation improvements and stream-style chunk helpers are added so the
utilities behave safely when used in incremental workflows.
"""

import numpy as np


def _as_non_empty_array(x, name):
    """
    Convert input to a non-empty NumPy array.
    """
    x = np.asarray(x)

    if x.size == 0:
        raise ValueError(f"{name} cannot be empty")

    return x


def _validate_same_shape(a, b):
    """
    Check that two vectors have the same shape.
    """
    if a.shape != b.shape:
        raise ValueError("input vectors must have the same shape")


def logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    """
    x = _as_non_empty_array(x, "x")
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
    x = _as_non_empty_array(x, "x")
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
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    _validate_same_shape(a, b)

    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(a, b):
    """
    Compute Manhattan distance between two vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    _validate_same_shape(a, b)

    return float(np.sum(np.abs(a - b)))


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    _validate_same_shape(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero-norm vectors.")

    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_euclidean(X, Y):
    """
    Compute pairwise Euclidean distances between two sets of vectors.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2D arrays")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T

    distances_squared = X_norm + Y_norm - 2 * np.dot(X, Y.T)

    return np.sqrt(np.maximum(distances_squared, 0))


def topk_indices(values, k, largest=True):
    """
    Return indices of the top-k values without full sorting.
    """
    values = _as_non_empty_array(values, "values")

    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError("k must be an integer")

    n = values.size

    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and {n}, got {k}.")

    if largest:
        return np.argpartition(values, n - k)[n - k:]

    return np.argpartition(values, k - 1)[:k]


def batch_iter(X, batch_size, y=None, shuffle=False, seed=None):
    """
    Yield mini-batches from X and optionally y.
    """
    X = np.asarray(X)

    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample")

    if not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, bool):
        raise ValueError("batch_size must be an integer")

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    n_samples = X.shape[0]

    if y is not None:
        y = np.asarray(y)

        if y.shape[0] != n_samples:
            raise ValueError("X and y must contain the same number of samples")

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        batch_indices = indices[start:start + batch_size]

        if y is not None:
            yield X[batch_indices], y[batch_indices]
        else:
            yield X[batch_indices]


def chunk_iter(X, chunk_size, y=None, shuffle=False, seed=None):
    """
    Yield chunks from X and optionally y.

    This is a stream-friendly alias around batch_iter. It is named separately
    because chunks are commonly used to simulate incoming data streams.
    """
    yield from batch_iter(
        X,
        batch_size=chunk_size,
        y=y,
        shuffle=shuffle,
        seed=seed,
    )