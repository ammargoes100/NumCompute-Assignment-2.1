"""
Tests for retained benchmarking utilities.
"""

import numpy as np

from numcompute_stream.benchmarking import (
    timeit,
    compare,
    run_vectorisation_benchmarks,
)


def test_timeit_returns_correct_keys():
    result = timeit(np.sum, np.array([1, 2, 3]), repeats=3)

    for key in ("mean", "std", "min", "max", "runs"):
        assert key in result


def test_timeit_correct_number_of_runs():
    result = timeit(np.sum, np.array([1, 2, 3]), repeats=7)

    assert len(result["runs"]) == 7


def test_compare_speedup_is_positive():
    rng = np.random.default_rng(42)
    a = rng.random(1000)
    b = rng.random(1000)

    def loop_dot(x, y):
        total = 0.0
        for i in range(len(x)):
            total += x[i] * y[i]
        return total

    def vec_dot(x, y):
        return np.dot(x, y)

    result = compare("dot", loop_dot, vec_dot, a, b, repeats=3)

    assert result["speedup"] > 0


def test_run_vectorisation_benchmarks_returns_five_results():
    results = run_vectorisation_benchmarks(
        n=100,
        repeats=2,
        seed=42,
        print_results=False,
    )

    assert isinstance(results, list)
    assert len(results) == 5