"""
Tests for visualisation utilities.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")

from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)


def test_plot_metric_over_time_returns_figure_and_axes():
    values = np.array([0.5, 0.6, 0.8])

    fig, ax = plot_metric_over_time(values, title="Accuracy", ylabel="Accuracy")

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Accuracy"
    assert ax.get_ylabel() == "Accuracy"


def test_compare_models_returns_figure_and_axes():
    metric1 = np.array([0.5, 0.6, 0.7])
    metric2 = np.array([0.4, 0.65, 0.8])

    fig, ax = compare_models(metric1, metric2, labels=("Tree", "Ensemble"))

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Model comparison over stream chunks"


def test_plot_predictions_vs_ground_truth_returns_figure_and_axes():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    fig, ax = plot_predictions_vs_ground_truth(y_true, y_pred)

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Predictions vs ground truth"


def test_plot_metric_over_time_has_correct_number_of_points():
    values = np.array([0.5, 0.6, 0.8])

    fig, ax = plot_metric_over_time(values)

    line = ax.lines[0]

    assert len(line.get_xdata()) == 3
    assert len(line.get_ydata()) == 3


def test_compare_models_has_two_lines():
    metric1 = np.array([0.5, 0.6])
    metric2 = np.array([0.4, 0.7])

    fig, ax = compare_models(metric1, metric2)

    assert len(ax.lines) == 2
def test_plot_metric_over_time_rejects_empty_values():
    import pytest

    with pytest.raises(ValueError):
        plot_metric_over_time(np.array([]))


def test_plot_metric_over_time_rejects_nan_values():
    import pytest

    with pytest.raises(ValueError):
        plot_metric_over_time(np.array([0.5, np.nan]))


def test_compare_models_rejects_bad_labels():
    import pytest

    with pytest.raises(ValueError):
        compare_models(np.array([0.5]), np.array([0.6]), labels=("Only one",))


def test_compare_models_custom_title_and_ylabel():
    metric1 = np.array([0.5, 0.6])
    metric2 = np.array([0.4, 0.7])

    fig, ax = compare_models(
        metric1,
        metric2,
        labels=("Tree", "Ensemble"),
        title="Tree vs Ensemble",
        ylabel="Accuracy",
    )

    assert ax.get_title() == "Tree vs Ensemble"
    assert ax.get_ylabel() == "Accuracy"


def test_predictions_plot_rejects_shape_mismatch():
    import pytest

    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth(
            np.array([0, 1]),
            np.array([0, 1, 1]),
        )


def test_predictions_plot_rejects_empty_input():
    import pytest

    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth(np.array([]), np.array([]))


def test_plot_metric_can_save_file(tmp_path):
    save_path = tmp_path / "metric.png"

    plot_metric_over_time(np.array([0.5, 0.6]), save_path=save_path)

    assert save_path.exists()


def test_compare_models_can_save_file(tmp_path):
    save_path = tmp_path / "compare.png"

    compare_models(
        np.array([0.5, 0.6]),
        np.array([0.4, 0.7]),
        save_path=save_path,
    )

    assert save_path.exists()


def test_predictions_plot_can_save_file(tmp_path):
    save_path = tmp_path / "predictions.png"

    plot_predictions_vs_ground_truth(
        np.array([0, 1]),
        np.array([0, 1]),
        save_path=save_path,
    )

    assert save_path.exists()