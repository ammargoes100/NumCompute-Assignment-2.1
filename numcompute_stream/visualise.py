"""
Visualisation utilities for NumCompute-Stream.

This module provides lightweight matplotlib plotting helpers for streaming
machine learning experiments. The functions are designed for notebooks,
scripts, demo outputs, and saved benchmark figures.
"""

import numpy as np
import matplotlib.pyplot as plt


def _validate_1d_numeric(values, name):
    """
    Convert values to a non-empty 1D numeric array.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")

    if values.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if np.isnan(values).any():
        raise ValueError(f"{name} cannot contain NaN values")

    return values


def _validate_labels(labels):
    """
    Validate model comparison labels.
    """
    if not isinstance(labels, (list, tuple)) or len(labels) != 2:
        raise ValueError("labels must contain exactly two model names")

    return labels


def _finalise_plot(fig, save_path=None, show=False):
    """
    Save and/or show a matplotlib figure.
    """
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_metric_over_time(
    metric_values,
    title="Metric over time",
    ylabel="Metric",
    save_path=None,
    show=False,
):
    """
    Plot one metric across stream chunks.

    Parameters
    ----------
    metric_values : array-like
        Metric values recorded over chunks.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str or None, default=None
        If provided, save the plot to this path.
    show : bool, default=False
        Whether to display the plot immediately.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    metric_values = _validate_1d_numeric(metric_values, "metric_values")
    chunks = np.arange(1, metric_values.size + 1)

    fig, ax = plt.subplots()
    ax.plot(chunks, metric_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    _finalise_plot(fig, save_path=save_path, show=show)

    return fig, ax


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title="Model comparison over stream chunks",
    ylabel="Metric",
    save_path=None,
    show=False,
):
    """
    Compare two models using metric values over stream chunks.

    Parameters
    ----------
    metric1 : array-like
        Metric values for the first model.
    metric2 : array-like
        Metric values for the second model.
    labels : tuple of str
        Labels for the two models.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str or None, default=None
        If provided, save the plot to this path.
    show : bool, default=False
        Whether to display the plot immediately.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    metric1 = _validate_1d_numeric(metric1, "metric1")
    metric2 = _validate_1d_numeric(metric2, "metric2")
    labels = _validate_labels(labels)

    chunks1 = np.arange(1, metric1.size + 1)
    chunks2 = np.arange(1, metric2.size + 1)

    fig, ax = plt.subplots()
    ax.plot(chunks1, metric1, marker="o", label=labels[0])
    ax.plot(chunks2, metric2, marker="o", label=labels[1])

    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalise_plot(fig, save_path=save_path, show=show)

    return fig, ax


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title="Predictions vs ground truth",
    save_path=None,
    show=False,
):
    """
    Visualise predictions against ground truth labels for one chunk.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    title : str
        Plot title.
    save_path : str or None, default=None
        If provided, save the plot to this path.
    show : bool, default=False
        Whether to display the plot immediately.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred cannot be empty")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    sample_index = np.arange(y_true.size)

    fig, ax = plt.subplots()
    ax.scatter(sample_index, y_true, label="Ground truth", marker="o")
    ax.scatter(sample_index, y_pred, label="Prediction", marker="x")

    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class label")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalise_plot(fig, save_path=save_path, show=show)

    return fig, ax