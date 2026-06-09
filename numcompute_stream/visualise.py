"""
Visualisation utilities for NumCompute-Stream.

This module provides lightweight matplotlib plotting helpers for streaming
machine learning experiments. The functions are designed for notebooks,
scripts, and demo outputs.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric"):
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

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    metric_values = np.asarray(metric_values, dtype=float)
    chunks = np.arange(1, metric_values.size + 1)

    fig, ax = plt.subplots()
    ax.plot(chunks, metric_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    return fig, ax


def compare_models(metric1, metric2, labels=("Model 1", "Model 2")):
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

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    metric1 = np.asarray(metric1, dtype=float)
    metric2 = np.asarray(metric2, dtype=float)

    chunks = np.arange(1, max(metric1.size, metric2.size) + 1)

    fig, ax = plt.subplots()
    ax.plot(chunks[:metric1.size], metric1, marker="o", label=labels[0])
    ax.plot(chunks[:metric2.size], metric2, marker="o", label=labels[1])

    ax.set_title("Model comparison over stream chunks")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Metric")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred):
    """
    Visualise predictions against ground truth labels for one chunk.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    sample_index = np.arange(y_true.size)

    fig, ax = plt.subplots()
    ax.scatter(sample_index, y_true, label="Ground truth", marker="o")
    ax.scatter(sample_index, y_pred, label="Prediction", marker="x")

    ax.set_title("Predictions vs ground truth")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class label")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax