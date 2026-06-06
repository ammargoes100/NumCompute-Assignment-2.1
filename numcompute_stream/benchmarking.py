"""
Benchmarking utilities for NumCompute-Stream.

This module starts from the benchmarking tools in the original NumCompute
package. The basic timing helpers and loop-vs-NumPy comparisons are retained
because they are still useful for checking performance of numerical code.
"""

import time
import numpy as np


# ---------------------------------------------------------------------
# General timing utilities retained from the original package
# ---------------------------------------------------------------------

def timeit(fn, *args, repeats=10, **kwargs):
    """
    Time a callable over multiple runs.
    """
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.asarray(times, dtype=float)

    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "runs": times.tolist(),
    }


def compare(label, loop_fn, vec_fn, *args, repeats=10, **kwargs):
    """
    Compare a Python-loop implementation against a vectorised implementation.
    """
    loop_stats = timeit(loop_fn, *args, repeats=repeats, **kwargs)
    vec_stats = timeit(vec_fn, *args, repeats=repeats, **kwargs)

    speedup = (
        loop_stats["mean"] / vec_stats["mean"]
        if vec_stats["mean"] > 0
        else float("inf")
    )

    return {
        "label": label,
        "loop": loop_stats,
        "vectorised": vec_stats,
        "speedup": float(speedup),
    }


def print_vectorisation_table(results):
    """
    Print a formatted loop-vs-vectorised benchmark table.
    """
    col = "{:<28} {:>12} {:>12} {:>10}"

    print("\n" + "=" * 66)
    print(col.format("Operation", "Loop (ms)", "Vec (ms)", "Speedup"))
    print("=" * 66)

    for result in results:
        loop_ms = result["loop"]["mean"] * 1000
        vec_ms = result["vectorised"]["mean"] * 1000

        print(
            col.format(
                result["label"],
                f"{loop_ms:.3f}",
                f"{vec_ms:.3f}",
                f"{result['speedup']:.1f}x",
            )
        )

    print("=" * 66 + "\n")


# ---------------------------------------------------------------------
# Small loop-vs-NumPy examples retained from the original package
# ---------------------------------------------------------------------

def _dot_loop(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def _dot_vec(a, b):
    return np.dot(a, b)


def _rowmean_loop(X):
    out = []
    for i in range(X.shape[0]):
        row_sum = 0.0
        for j in range(X.shape[1]):
            row_sum += X[i, j]
        out.append(row_sum / X.shape[1])
    return out


def _rowmean_vec(X):
    return np.mean(X, axis=1)


def _square_loop(x):
    out = []
    for value in x:
        out.append(value * value)
    return out


def _square_vec(x):
    return x ** 2


def _euclid_loop(a, b):
    total = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        total += diff * diff
    return total ** 0.5


def _euclid_vec(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def _softmax_loop(x):
    """
    Stable softmax using a loop-style implementation.
    """
    max_value = max(x)
    exps = [np.exp(value - max_value) for value in x]
    total = sum(exps)
    return [value / total for value in exps]


def _softmax_vec(x):
    """
    Stable softmax using a vectorised NumPy implementation.
    """
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_vectorisation_benchmarks(n=10_000, repeats=20, seed=42, print_results=True):
    """
    Run loop-vs-vectorised benchmark examples.
    """
    rng = np.random.default_rng(seed)

    a = rng.random(n)
    b = rng.random(n)
    X = rng.random((n // 10, 10))

    results = [
        compare("dot product", _dot_loop, _dot_vec, a, b, repeats=repeats),
        compare("element-wise square", _square_loop, _square_vec, a, repeats=repeats),
        compare("euclidean distance", _euclid_loop, _euclid_vec, a, b, repeats=repeats),
        compare("row-wise mean", _rowmean_loop, _rowmean_vec, X, repeats=repeats),
        compare("softmax", _softmax_loop, _softmax_vec, a, repeats=repeats),
    ]

    if print_results:
        print(f"Vectorisation benchmarks: n={n}, repeats={repeats}, seed={seed}")
        print_vectorisation_table(results)

    return results


def run_all(n=10_000, repeats=20, seed=42):
    """
    Backward-compatible alias for the original benchmark runner.
    """
    return run_vectorisation_benchmarks(n=n, repeats=repeats, seed=seed)


if __name__ == "__main__":
    run_vectorisation_benchmarks()