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
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")

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

    The max shift is used so very large input values do not overflow.
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("softmax input cannot be empty")

    max_value = float(np.max(x))
    exps = [np.exp(value - max_value) for value in x]
    total = sum(exps)

    if total == 0:
        return [1.0 / len(x)] * len(x)

    return [value / total for value in exps]


def _softmax_vec(x):
    """
    Stable softmax using a vectorised NumPy implementation.
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("softmax input cannot be empty")

    shifted = x - np.max(x)
    exp_values = np.exp(shifted)
    total = np.sum(exp_values)

    if total == 0:
        return np.full_like(x, 1.0 / x.size, dtype=float)

    return exp_values / total


def run_vectorisation_benchmarks(n=10_000, repeats=20, seed=42, print_results=True):
    """
    Run loop-vs-vectorised benchmark examples.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)

    a = rng.random(n)
    b = rng.random(n)
    X = rng.random((max(1, n // 10), 10))

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

# ---------------------------------------------------------------------
# Streaming model benchmarking utilities
# ---------------------------------------------------------------------

def _accuracy(y_true, y_pred):
    """
    Compute simple classification accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        return 0.0

    return float(np.mean(y_true == y_pred))


def benchmark_streaming_model(model, chunks, name="model"):
    """
    Benchmark one streaming model over a sequence of chunks.

    The model must implement partial_fit(X, y) and predict(X).
    """
    if not hasattr(model, "partial_fit"):
        raise AttributeError("model must implement partial_fit(X, y)")

    if not hasattr(model, "predict"):
        raise AttributeError("model must implement predict(X)")

    fit_times = []
    predict_times = []
    accuracies = []

    total_correct = 0
    total_seen = 0
    chunk_count = 0

    for X_chunk, y_chunk in chunks:
        X_chunk = np.asarray(X_chunk)
        y_chunk = np.asarray(y_chunk)

        if X_chunk.shape[0] != y_chunk.shape[0]:
            raise ValueError("X_chunk and y_chunk must contain the same number of samples")

        start_fit = time.perf_counter()
        model.partial_fit(X_chunk, y_chunk)
        fit_times.append(time.perf_counter() - start_fit)

        start_predict = time.perf_counter()
        y_pred = model.predict(X_chunk)
        predict_times.append(time.perf_counter() - start_predict)

        chunk_accuracy = _accuracy(y_chunk, y_pred)
        accuracies.append(chunk_accuracy)

        total_correct += int(np.sum(y_chunk == y_pred))
        total_seen += int(y_chunk.size)
        chunk_count += 1

    if chunk_count == 0:
        raise ValueError("chunks must contain at least one chunk")

    fit_times = np.asarray(fit_times, dtype=float)
    predict_times = np.asarray(predict_times, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)

    final_accuracy = total_correct / total_seen if total_seen > 0 else 0.0

    return {
        "name": name,
        "chunks": chunk_count,
        "fit_times": fit_times.tolist(),
        "predict_times": predict_times.tolist(),
        "accuracies": accuracies.tolist(),
        "mean_fit_time": float(np.mean(fit_times)),
        "mean_predict_time": float(np.mean(predict_times)),
        "final_accuracy": float(final_accuracy),
    }


def compare_streaming_models(models, chunks):
    """
    Benchmark multiple streaming models on the same data chunks.

    The chunks are converted to a list so each model receives the same stream.
    """
    chunks = list(chunks)

    if len(chunks) == 0:
        raise ValueError("chunks must contain at least one chunk")

    results = []

    for name, model in models.items():
        result = benchmark_streaming_model(model, chunks, name=name)
        results.append(result)

    return results


def print_streaming_table(results):
    """
    Print a formatted streaming model benchmark table.
    """
    col = "{:<24} {:>8} {:>16} {:>16} {:>14}"

    print("\n" + "=" * 86)
    print(
        col.format(
            "Model",
            "Chunks",
            "Fit/chunk (ms)",
            "Pred/chunk (ms)",
            "Final Acc",
        )
    )
    print("=" * 86)

    for result in results:
        fit_ms = result["mean_fit_time"] * 1000
        pred_ms = result["mean_predict_time"] * 1000

        print(
            col.format(
                result["name"],
                result["chunks"],
                f"{fit_ms:.3f}",
                f"{pred_ms:.3f}",
                f"{result['final_accuracy']:.3f}",
            )
        )

    print("=" * 86 + "\n")


def make_stream_chunks(X, y, chunk_size):
    """
    Split arrays into smaller chunks to simulate a data stream.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    for start in range(0, X.shape[0], chunk_size):
        end = start + chunk_size
        yield X[start:end], y[start:end]

if __name__ == "__main__":
    run_vectorisation_benchmarks()