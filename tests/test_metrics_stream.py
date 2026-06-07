"""
Tests for retained metric functions.
"""

import numpy as np
import pytest

from numcompute_stream.metrics import (
    accuracy,
    precision,
    recall,
    f1,
    confusion_matrix,
    mse,
    roc_curve,
    auc,
    StreamingAccuracy,
    StreamingConfusionMatrix,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingMSE,
)


def test_accuracy_basic():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    assert accuracy(y_true, y_pred) == 0.8


def test_accuracy_all_correct():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])

    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_shape_mismatch():
    with pytest.raises(ValueError):
        accuracy(np.array([1, 0]), np.array([1, 0, 1]))


def test_confusion_matrix_basic():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    expected = np.array([
        [2, 0],
        [1, 2],
    ])

    assert np.array_equal(confusion_matrix(y_true, y_pred), expected)


def test_confusion_matrix_with_string_labels():
    y_true = np.array(["cat", "dog", "cat"])
    y_pred = np.array(["cat", "cat", "cat"])

    expected = np.array([
        [2, 0],
        [1, 0],
    ])

    result = confusion_matrix(y_true, y_pred, labels=["cat", "dog"])

    assert np.array_equal(result, expected)


def test_precision_binary():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    assert precision(y_true, y_pred) == 1.0


def test_precision_zero_division():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0, 0, 0])

    assert precision(y_true, y_pred) == 0.0


def test_precision_macro():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])

    expected = (1.0 + 0.5 + 1.0) / 3

    assert np.isclose(precision(y_true, y_pred, average="macro"), expected)


def test_recall_binary():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    assert np.isclose(recall(y_true, y_pred), 2 / 3)


def test_recall_zero_division():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 1, 0])

    assert recall(y_true, y_pred) == 0.0


def test_recall_macro():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])

    expected = (1.0 + 1.0 + 0.5) / 3

    assert np.isclose(recall(y_true, y_pred, average="macro"), expected)


def test_f1_binary():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    assert np.isclose(f1(y_true, y_pred), 0.8)


def test_f1_zero_division():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])

    assert f1(y_true, y_pred) == 0.0


def test_f1_macro():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])

    f1_class_0 = 1.0
    f1_class_1 = 2 * 0.5 * 1.0 / (0.5 + 1.0)
    f1_class_2 = 2 * 1.0 * 0.5 / (1.0 + 0.5)

    expected = (f1_class_0 + f1_class_1 + f1_class_2) / 3

    assert np.isclose(f1(y_true, y_pred, average="macro"), expected)


def test_mse_basic():
    y_true = np.array([3.0, 2.5, 4.0])
    y_pred = np.array([2.5, 2.0, 4.5])

    assert mse(y_true, y_pred) == 0.25


def test_mse_zero():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    assert mse(y_true, y_pred) == 0.0


def test_mse_non_numeric():
    with pytest.raises(ValueError):
        mse(np.array(["a", "b"]), np.array(["a", "c"]))


def test_roc_curve_basic():
    y_true = np.array([1, 0, 1, 1, 0])
    scores = np.array([0.9, 0.2, 0.4, 0.8, 0.1])

    fpr_values, tpr_values, thresholds = roc_curve(y_true, scores)

    assert fpr_values[0] == 0.0
    assert tpr_values[0] == 0.0
    assert thresholds[0] == np.inf


def test_roc_curve_one_class_error():
    y_true = np.array([1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7])

    with pytest.raises(ValueError):
        roc_curve(y_true, scores)


def test_auc_basic():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 1.0, 0.0])

    assert np.isclose(auc(x, y), 0.5)
def test_streaming_accuracy_accumulates_chunks():
    metric = StreamingAccuracy()

    metric.update(np.array([1, 0, 1]), np.array([1, 0, 0]))
    metric.update(np.array([1, 0]), np.array([1, 0]))

    assert metric.result() == 0.8


def test_streaming_accuracy_reset():
    metric = StreamingAccuracy()
    metric.update(np.array([1, 0]), np.array([1, 1]))

    metric.reset()

    assert metric.result() == 0.0


def test_streaming_confusion_matrix_accumulates_chunks():
    metric = StreamingConfusionMatrix(labels=[0, 1])

    metric.update(np.array([0, 1]), np.array([0, 1]))
    metric.update(np.array([1, 0]), np.array([0, 0]))

    expected = np.array([
        [2, 0],
        [1, 1],
    ])

    assert np.array_equal(metric.result(), expected)


def test_streaming_precision_matches_batch_precision():
    y_true_1 = np.array([1, 0, 1])
    y_pred_1 = np.array([1, 0, 0])
    y_true_2 = np.array([1, 0])
    y_pred_2 = np.array([1, 0])

    metric = StreamingPrecision()
    metric.update(y_true_1, y_pred_1)
    metric.update(y_true_2, y_pred_2)

    y_true_all = np.concatenate([y_true_1, y_true_2])
    y_pred_all = np.concatenate([y_pred_1, y_pred_2])

    assert metric.result() == precision(y_true_all, y_pred_all)


def test_streaming_recall_matches_batch_recall():
    y_true_1 = np.array([1, 0, 1])
    y_pred_1 = np.array([1, 0, 0])
    y_true_2 = np.array([1, 0])
    y_pred_2 = np.array([1, 0])

    metric = StreamingRecall()
    metric.update(y_true_1, y_pred_1)
    metric.update(y_true_2, y_pred_2)

    y_true_all = np.concatenate([y_true_1, y_true_2])
    y_pred_all = np.concatenate([y_pred_1, y_pred_2])

    assert metric.result() == recall(y_true_all, y_pred_all)


def test_streaming_f1_matches_batch_f1():
    y_true_1 = np.array([1, 0, 1])
    y_pred_1 = np.array([1, 0, 0])
    y_true_2 = np.array([1, 0])
    y_pred_2 = np.array([1, 0])

    metric = StreamingF1()
    metric.update(y_true_1, y_pred_1)
    metric.update(y_true_2, y_pred_2)

    y_true_all = np.concatenate([y_true_1, y_true_2])
    y_pred_all = np.concatenate([y_pred_1, y_pred_2])

    assert metric.result() == f1(y_true_all, y_pred_all)


def test_streaming_mse_accumulates_chunks():
    metric = StreamingMSE()

    metric.update(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    metric.update(np.array([4.0, 5.0]), np.array([6.0, 5.0]))

    expected = (0.0 + 1.0 + 4.0 + 0.0) / 4

    assert metric.result() == expected


def test_streaming_mse_reset():
    metric = StreamingMSE()
    metric.update(np.array([1.0, 2.0]), np.array([2.0, 3.0]))

    metric.reset()

    assert metric.result() == 0.0