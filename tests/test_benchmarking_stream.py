"""
Tests for numcompute_stream.benchmarking.
"""

import numpy as np
import pytest

from numcompute_stream.benchmarking import (
    timeit,
    compare,
    run_vectorisation_benchmarks,
    benchmark_streaming_model,
    compare_streaming_models,
    make_stream_chunks,
)


class DummyStreamingClassifier:
    """
    Simple streaming classifier used only for benchmarking tests.

    It memorises the majority class seen so far and predicts that class.
    """

    def __init__(self):
        self.classes_seen = []
        self.majority_class = 0

    def partial_fit(self, X, y):
        y = np.asarray(y)
        self.classes_seen.extend(y.tolist())

        values, counts = np.unique(np.asarray(self.classes_seen), return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.full(X.shape[0], self.majority_class)


def test_timeit_returns_correct_keys():
    result = timeit(np.sum, np.array([1, 2, 3]), repeats=3)

    for key in ("mean", "std", "min", "max", "runs"):
        assert key in result


def test_timeit_correct_number_of_runs():
    result = timeit(np.sum, np.array([1, 2, 3]), repeats=7)

    assert len(result["runs"]) == 7


def test_timeit_rejects_zero_repeats():
    with pytest.raises(ValueError):
        timeit(np.sum, np.array([1, 2, 3]), repeats=0)


def test_timeit_rejects_negative_repeats():
    with pytest.raises(ValueError):
        timeit(np.sum, np.array([1, 2, 3]), repeats=-2)


def test_compare_returns_expected_keys():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    def loop_dot(x, y):
        total = 0.0
        for i in range(len(x)):
            total += x[i] * y[i]
        return total

    def vec_dot(x, y):
        return np.dot(x, y)

    result = compare("dot", loop_dot, vec_dot, a, b, repeats=2)

    assert result["label"] == "dot"
    assert "loop" in result
    assert "vectorised" in result
    assert "speedup" in result


def test_compare_speedup_is_positive():
    a = np.random.default_rng(42).random(1000)
    b = np.random.default_rng(123).random(1000)

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


def test_run_vectorisation_benchmarks_rejects_invalid_n():
    with pytest.raises(ValueError):
        run_vectorisation_benchmarks(n=0, repeats=2, print_results=False)


def test_make_stream_chunks_correct_chunk_sizes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    chunks = list(make_stream_chunks(X, y, chunk_size=4))

    assert len(chunks) == 3
    assert chunks[0][0].shape == (4, 2)
    assert chunks[1][0].shape == (4, 2)
    assert chunks[2][0].shape == (2, 2)


def test_make_stream_chunks_rejects_mismatched_lengths():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(9)

    with pytest.raises(ValueError):
        list(make_stream_chunks(X, y, chunk_size=4))


def test_make_stream_chunks_rejects_invalid_chunk_size():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    with pytest.raises(ValueError):
        list(make_stream_chunks(X, y, chunk_size=0))


def test_benchmark_streaming_model_returns_expected_keys():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = np.array([0, 1] * 10)

    chunks = list(make_stream_chunks(X, y, chunk_size=5))
    model = DummyStreamingClassifier()

    result = benchmark_streaming_model(model, chunks, name="dummy")

    expected_keys = {
        "name",
        "chunks",
        "fit_times",
        "predict_times",
        "accuracies",
        "mean_fit_time",
        "mean_predict_time",
        "final_accuracy",
    }

    assert expected_keys.issubset(result.keys())
    assert result["name"] == "dummy"
    assert result["chunks"] == 4


def test_benchmark_streaming_model_accuracy_in_valid_range():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 2))
    y = np.array([0, 1, 1] * 10)

    chunks = list(make_stream_chunks(X, y, chunk_size=10))
    model = DummyStreamingClassifier()

    result = benchmark_streaming_model(model, chunks)

    assert 0.0 <= result["final_accuracy"] <= 1.0


def test_benchmark_streaming_model_rejects_empty_chunks():
    model = DummyStreamingClassifier()

    with pytest.raises(ValueError):
        benchmark_streaming_model(model, [])


def test_benchmark_streaming_model_requires_partial_fit():
    class NoPartialFit:
        def predict(self, X):
            return np.zeros(X.shape[0])

    X = np.ones((5, 2))
    y = np.zeros(5)
    chunks = [(X, y)]

    with pytest.raises(AttributeError):
        benchmark_streaming_model(NoPartialFit(), chunks)


def test_benchmark_streaming_model_requires_predict():
    class NoPredict:
        def partial_fit(self, X, y):
            return self

    X = np.ones((5, 2))
    y = np.zeros(5)
    chunks = [(X, y)]

    with pytest.raises(AttributeError):
        benchmark_streaming_model(NoPredict(), chunks)


def test_compare_streaming_models_returns_one_result_per_model():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 2))
    y = np.array([0, 1] * 10)

    chunks = list(make_stream_chunks(X, y, chunk_size=5))

    models = {
        "dummy_a": DummyStreamingClassifier(),
        "dummy_b": DummyStreamingClassifier(),
    }

    results = compare_streaming_models(models, chunks)

    assert len(results) == 2
    assert results[0]["name"] == "dummy_a"
    assert results[1]["name"] == "dummy_b"