"""
Tests for benchmarking.py
"""
import numpy as np
import pytest
from numcompute.benchmarking import timeit, compare, run_all


def test_timeit_returns_correct_keys():
    # make sure timeit gives back all the stats we expect
    result = timeit(np.sum, np.array([1, 2, 3]), repeats=3)
    for key in ('mean', 'std', 'min', 'max', 'runs'):
        assert key in result


def test_timeit_correct_number_of_runs():
    # runs list should have exactly as many entries as repeats
    result = timeit(np.sum, np.array([1, 2, 3]), repeats=7)
    assert len(result['runs']) == 7


def test_compare_speedup_is_positive():
    # speedup should always be a positive number
    a = np.random.rand(1000)
    b = np.random.rand(1000)

    def loop_dot(a, b):
        total = 0.0
        for i in range(len(a)):
            total += a[i] * b[i]
        return total

    def vec_dot(a, b):
        return np.dot(a, b)

    result = compare("dot", loop_dot, vec_dot, a, b, repeats=5)
    assert result['speedup'] > 0