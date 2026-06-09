"""
Streaming benchmark for NumCompute-Stream.

This script compares a single DecisionTreeClassifier with an EnsembleClassifier
under a chunked streaming learning scenario.

Run from the project root:

    python benchmarks/stream_benchmark.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from numcompute_stream.io import load_csv
from numcompute_stream.benchmarking import make_stream_chunks
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.visualise import compare_models, plot_metric_over_time


def run_model(name, model, chunks):
    """
    Train and evaluate one streaming model.
    """
    trainer = StreamTrainer(model)
    trainer.fit_stream(chunks)

    final_cumulative_accuracy = trainer.get_history("cumulative_accuracy")[-1]
    mean_chunk_accuracy = float(np.mean(trainer.get_history("score")))
    total_fit_time = float(np.sum(trainer.get_history("fit_time")))
    total_predict_time = float(np.sum(trainer.get_history("predict_time")))
    final_memory_bytes = trainer.get_history("memory_bytes")[-1]

    summary = {
        "name": name,
        "final_cumulative_accuracy": final_cumulative_accuracy,
        "mean_chunk_accuracy": mean_chunk_accuracy,
        "total_fit_time": total_fit_time,
        "total_predict_time": total_predict_time,
        "final_memory_bytes": final_memory_bytes,
        "history": trainer.get_history(),
        "cumulative_accuracy": trainer.get_history("cumulative_accuracy"),
        "chunk_accuracy": trainer.get_history("score"),
    }

    return summary


def print_summary(summary):
    """
    Print a compact benchmark summary.
    """
    print(f"\n{summary['name']}")
    print("-" * len(summary["name"]))
    print(f"Final cumulative accuracy: {summary['final_cumulative_accuracy']:.4f}")
    print(f"Mean chunk accuracy:       {summary['mean_chunk_accuracy']:.4f}")
    print(f"Total fit time:           {summary['total_fit_time']:.6f} seconds")
    print(f"Total predict time:       {summary['total_predict_time']:.6f} seconds")
    print(f"Final memory estimate:    {summary['final_memory_bytes']} bytes")


def main():
    data_path = PROJECT_ROOT / "demo" / "data" / "diabetes.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Place diabetes.csv in demo/data first."
        )

    data, columns = load_csv(str(data_path), skip_header=True)

    X = data[:, :-1]
    y = data[:, -1].astype(int)

    rng = np.random.default_rng(42)
    indices = rng.permutation(X.shape[0])

    X_stream = X[indices]
    y_stream = y[indices]

    chunk_size = 64
    chunks = list(make_stream_chunks(X_stream, y_stream, chunk_size=chunk_size))

    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=5,
        criterion="gini",
        random_state=42,
    )

    ensemble = EnsembleClassifier(
        n_estimators=7,
        max_depth=4,
        min_samples_split=5,
        criterion="gini",
        max_features=4,
        bootstrap=True,
        random_state=42,
    )

    tree_summary = run_model("Decision Tree", tree, chunks)
    ensemble_summary = run_model("Ensemble Classifier", ensemble, chunks)

    print("NumCompute-Stream Benchmark")
    print("===========================")
    print(f"Dataset: {data_path}")
    print(f"Columns: {columns}")
    print(f"Dataset shape: {data.shape}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {len(chunks)}")

    print_summary(tree_summary)
    print_summary(ensemble_summary)

    output_dir = PROJECT_ROOT / "benchmarks" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_over_time(
        tree_summary["cumulative_accuracy"],
        title="Decision tree cumulative accuracy",
        ylabel="Cumulative accuracy",
        save_path=output_dir / "tree_cumulative_accuracy.png",
    )

    plot_metric_over_time(
        ensemble_summary["cumulative_accuracy"],
        title="Ensemble cumulative accuracy",
        ylabel="Cumulative accuracy",
        save_path=output_dir / "ensemble_cumulative_accuracy.png",
    )

    compare_models(
        tree_summary["cumulative_accuracy"],
        ensemble_summary["cumulative_accuracy"],
        labels=("Decision tree", "Ensemble"),
        title="Tree vs ensemble cumulative accuracy",
        ylabel="Cumulative accuracy",
        save_path=output_dir / "tree_vs_ensemble_accuracy.png",
    )

    print(f"\nSaved benchmark plots to: {output_dir}")


if __name__ == "__main__":
    main()
