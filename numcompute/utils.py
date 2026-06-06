"""
Module: utils.py
Description: Utility functions for the NumCompute toolkit. Includes numerically
             stable activations, distance metrics, and data batching helpers.
"""
import numpy as np


def logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    Parameters    x    : np.ndarray , axis : int or None
    Returns
    np.ndarray

    Time complexity  : O(n)
    Space complexity : O(n)
    """
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    result = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def softmax(x, axis=None):
    """
    Compute softmax in a numerically stable way.

    Parameters : x    : np.ndarray, axis : int or None
    Returns
    np.ndarray, same shape as x

    Time complexity  : O(n)
    Space complexity : O(n)
    """
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - m)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x):
    """
    Compute the sigmoid activation element-wise.
    Parameters x : np.ndarray
    Returns np.ndarray, same shape as x, values in (0, 1)

    Time complexity  : O(n)
    Space complexity : O(n)
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def relu(x):
    """
    Compute the ReLU activation element-wise.
    Parameters
    x : np.ndarray

    Returns
    np.ndarray, same shape as x

    Time complexity  : O(n)
    Space complexity : O(n)
    """
    return np.maximum(0, np.asarray(x, dtype=float))


def tanh(x):
    """
    Compute the tanh activation element-wise.

    Parameters
    x : np.ndarray
    Returns
    np.ndarray, same shape as x, values in (-1, 1)

    Time complexity  : O(n)
    Space complexity : O(n)
    """
    return np.tanh(np.asarray(x, dtype=float))


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors.
    Parameters
    a : np.ndarray, shape (n,)
    b : np.ndarray, shape (n,)
    Returns
    float
    Time complexity  : O(n)
    Space complexity : O(1)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.sum((a - b) ** 2))


def manhattan_distance(a, b):
    """
    Compute Manhattan (L1) distance between two vectors.

    Parameters
    a : np.ndarray, shape (n,)
    b : np.ndarray, shape (n,)

    Returns
    float

    Time complexity  : O(n)
    Space complexity : O(1)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b))


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.

    Parameters
    a : np.ndarray, shape (n,)
    b : np.ndarray, shape (n,)

    Returns
    float, in [-1, 1]

    Raises
    ValueError
        If either vector has zero norm.

    Time complexity  : O(n)
    Space complexity : O(1)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero-norm vectors.")
    return np.dot(a, b) / (norm_a * norm_b)


def pairwise_euclidean(X, Y):
    """
    Compute pairwise Euclidean distances between two sets of vectors.

    Parameters
    X : np.ndarray, shape (n, d)
    Y : np.ndarray, shape (m, d)

    Returns
    np.ndarray, shape (n, m)
        Entry (i, j) is the distance between X[i] and Y[j].

    Time complexity  : O(n * m * d)
    Space complexity : O(n * m)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T
    dist = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    return np.sqrt(np.maximum(dist, 0))


def topk_indices(values, k, largest=True):
    """
    Return indices of the top-k values without full sorting.

    Parameters
    values  : np.ndarray, shape (n,)
    k       : int, number of elements to select
    largest : bool, if True return largest k, else smallest k

    Returns
    np.ndarray, shape (k,)
        Indices of the top-k elements (unordered).

    Raises
    ValueError
        If k < 1 or k > len(values).

    Time complexity  : O(n)  average via np.argpartition
    Space complexity : O(k)
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
    Yield mini-batches from X (and optionally y).

    Parameters
    X          : np.ndarray, shape (n, ...)
    batch_size : int
    y          : np.ndarray, shape (n,), optional
    shuffle    : bool, whether to shuffle before batching
    seed       : int or None, random seed for reproducibility

    Yields

    X_batch : np.ndarray, shape (batch_size, ...)
    y_batch : np.ndarray, shape (batch_size,)  — only if y is provided

    Raises
    ValueError
        If batch_size < 1.

    Time complexity  : O(n) per full pass
    Space complexity : O(batch_size)
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
        idx = indices[start:start + batch_size]
        if y is not None:
            yield X[idx], np.asarray(y)[idx]
        else:
            yield X[idx]