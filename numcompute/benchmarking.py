"""
Module: benchmarking.py
Description: Benchmarking tools to compare vectorised NumPy implementations
             against plain Python loops. Part of the NumCompute toolkit.
"""

import time
import numpy as np


def timeit(fn, *args, repeats=10, **kwargs):
    """
    Time a function over multiple runs and return stats.

    Parameters
    fn      : callable
              Function to benchmark
    *args   : positional args passed to fn
    repeats : int, optional (default=10)
              How many times to call fn
    **kwargs: keyword args passed to fn

    Returns
    dict with keys: 'mean', 'std', 'min', 'max', 'runs'
    """
    times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        # store how long that run took
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        'mean': float(np.mean(times)),
        'std':  float(np.std(times)),
        'min':  float(np.min(times)),
        'max':  float(np.max(times)),
        'runs': times.tolist(),
    }


def compare(label, loop_fn, vec_fn, *args, repeats=10, **kwargs):
    """
    Run both a loop and vectorised version, return timing + speedup.
    Parameters
    label   : str
              Name shown in results table
    loop_fn : callable
              Python loop implementation
    vec_fn  : callable
              Vectorised NumPy implementation
    *args   : shared args passed to both functions
    repeats : int, optional (default=10)
    **kwargs: shared keyword args passed to both functions

    Returns
    dict with keys: 'label', 'loop', 'vectorised', 'speedup'
    """
    loop_stats = timeit(loop_fn, *args, repeats=repeats, **kwargs)
    vec_stats  = timeit(vec_fn,  *args, repeats=repeats, **kwargs)

    # how many times faster is the vectorised version
    speedup = loop_stats['mean'] / vec_stats['mean'] if vec_stats['mean'] > 0 else float('inf')

    return {
        'label':      label,
        'loop':       loop_stats,
        'vectorised': vec_stats,
        'speedup':    speedup,
    }


def print_table(results):
    """
    Print a formatted performance comparison table.
    Parameters
    results : list of dicts
              Output from one or more compare() calls
    """
    col = "{:<28} {:>12} {:>12} {:>10}"

    print("\n" + "=" * 66)
    print(col.format("Operation", "Loop (ms)", "Vec (ms)", "Speedup"))
    print("=" * 66)

    for r in results:
        # convert seconds to milliseconds for readability
        loop_ms = r['loop']['mean'] * 1000
        vec_ms  = r['vectorised']['mean'] * 1000
        print(col.format(
            r['label'],
            f"{loop_ms:.3f}",
            f"{vec_ms:.3f}",
            f"{r['speedup']:.1f}x",
        ))

    print("=" * 66 + "\n")


# loop implementations

def _dot_loop(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def _rowmean_loop(X):
    out = []
    for i in range(X.shape[0]):
        row_sum = 0.0
        for j in range(X.shape[1]):
            row_sum += X[i, j]
        out.append(row_sum / X.shape[1])
    return out

def _square_loop(x):
    out = []
    for val in x:
        out.append(val * val)
    return out

def _euclid_loop(a, b):
    total = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        total += diff * diff
    return total ** 0.5

def _softmax_loop(x):
    # shift by max to avoid overflow, same logic as the vectorised version
    m = max(x)
    exps = [np.exp(v - m) for v in x]
    total = sum(exps)
    return [e / total for e in exps]


# vectorised implementations 
def _dot_vec(a, b):
    return np.dot(a, b)

def _rowmean_vec(X):
    return np.mean(X, axis=1)

def _square_vec(x):
    return x ** 2

def _euclid_vec(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def _softmax_vec(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# main runner

def run_all(n=10_000, repeats=20, seed=42):
    """
    Run all benchmarks and print results table.

    Parameters
    n       : int, optional (default=10_000)
              Size of test arrays
    repeats : int, optional (default=20)
              Timing repetitions per benchmark
    seed    : int, optional (default=42)
              Random seed so results are reproducible

    Returns
    list of result dicts
    """
    rng = np.random.default_rng(seed)

    # make test data once and reuse across all benchmarks
    a = rng.random(n)
    b = rng.random(n)
    X = rng.random((n // 10, 10))  # 2d array for the row-mean test

    print(f"Benchmarking with n={n}, repeats={repeats}, seed={seed}")

    results = [
        compare("dot product",        _dot_loop,     _dot_vec,     a, b, repeats=repeats),
        compare("element-wise square", _square_loop,  _square_vec,  a,   repeats=repeats),
        compare("euclidean distance",  _euclid_loop,  _euclid_vec,  a, b, repeats=repeats),
        compare("row-wise mean",       _rowmean_loop, _rowmean_vec, X,   repeats=repeats),
        compare("softmax",             _softmax_loop, _softmax_vec, a,   repeats=repeats),
    ]

    print_table(results)
    return results


# run directly with: python benchmarking.py
if __name__ == "__main__":
    run_all()