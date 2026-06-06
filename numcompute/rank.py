"""
Module: rank.py
Here, in rank_py, statistical analysis is being done. Here, we are going to perform ranking
of numerical data. Numpy is being used here. Here, we have included percentile function also.
It is being used to calculate q-th percentile of dataset. We are using interpolation techniques
such as lower, higher, linear and midpoint.
"""
import numpy as np


def rank(d, method="average"):
    
   
    d = np.asarray(d)

    if d.ndim != 1:
        raise ValueError("1D Array is expected by rank")
    if d.size == 0:
        raise ValueError("It is not possible to operate on empty array.")
    if np.issubdtype(d.dtype, np.number) and np.isnan(d).any():
        raise ValueError("NaN value is not supported")
    if method not in {"average", "dense", "ordinal"}:
        raise ValueError("method should be one of {'average', 'dense', 'ordinal'}.")

    n = d.size
    o = np.argsort(d, kind="stable")
    sorted = d[o]

    if method == "ordinal":
        ranks = np.empty(n, dtype=int)
        ranks[o] = np.arange(1, n + 1)
        return ranks

    
    g_start = np.empty(n, dtype=bool)
    g_start[0] = True
    g_start[1:] = sorted[1:] != sorted[:-1]

    start = np.flatnonzero(g_start)
    end = np.r_[start[1:], n]

    sorted_ranks = np.empty(n, dtype=float if method == "average" else int)

    if method == "average":
        for s, e in zip(start, end):
            avg_rank = (s + 1 + e) / 2.0
            sorted_ranks[s:e] = avg_rank

    elif method == "dense":
        dense_rank_values = np.arange(1, start.size + 1)
        for r, s, e in zip(dense_rank_values, start, end):
            sorted_ranks[s:e] = r

    ranks = np.empty_like(sorted_ranks)
    ranks[o] = sorted_ranks
    return ranks


def percentile(d, q, interpolation="linear"):
    
    d = np.asarray(d)

    if d.ndim != 1:
        raise ValueError("percentile expects a 1D array.")
    if d.size == 0:
        raise ValueError("percentile cannot operate on an empty array.")
    if np.issubdtype(d.dtype, np.number) and np.isnan(d).any():
        raise ValueError("percentile does not support NaN values.")
    if q < 0 or q > 100:
        raise ValueError("q must be in the range [0, 100].")
    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError("interpolation must be one of {'linear', 'lower', 'higher', 'midpoint'}.")

    x = np.sort(d, kind="stable")
    n = x.size

    if n == 1:
        return x[0]

    pos = (q / 100.0) * (n - 1)
    low = int(np.floor(pos))
    upper = int(np.ceil(pos))

    if low == upper:
        return x[low]

    if interpolation == "lower":
        return x[low]
    if interpolation == "higher":
        return x[upper]
    if interpolation == "midpoint":
        return (x[low] + x[upper]) / 2.0

    
    w = pos - low
    return x[low] * (1.0 - w) + x[upper] * w


    print(rank(np.array([10, 20, 20, 40]), method="average"))
print(rank(np.array([30, 20, 20, 40]), method="dense"))
print(rank(np.array([30, 20, 20, 40]), method="ordinal"))

print(rank(np.array([50, 80, 20, 60]), method="dense"))
print(rank(np.array([50, 80, 20, 60]), method="ordinal"))


print(percentile(np.array([60, 50, 30, 40, 80]), 50))
print(percentile(np.array([30, 20, 30, 40]), 30, interpolation="lower"))
print(percentile(np.array([10, 20, 30, 40]), 30, interpolation="higher"))
print(percentile(np.array([10, 20, 30, 40]), 30, interpolation="midpoint"))
print(percentile(np.array([10, 20, 30, 40]), 30, interpolation="linear"))


print(percentile(np.array([60, 50, 30, 40, 80]), 50))
print(percentile(np.array([30, 20, 30, 40]), 40, interpolation="lower"))
print(percentile(np.array([10, 20, 30, 40]), 40, interpolation="higher"))
print(percentile(np.array([10, 20, 30, 40]), 40, interpolation="midpoint"))
print(percentile(np.array([10, 20, 30, 40]), 40, interpolation="linear"))







