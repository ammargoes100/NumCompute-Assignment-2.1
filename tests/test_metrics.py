import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    precision,
    recall,
    f1,
    confusion_matrix,
    mse,
    roc_curve,
    auc,
)


def test_accuracy_basic():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    assert accuracy(y_true, y_pred) == 0.8


def test_accuracy_allcorrect():
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


def test_confusion_matrix():
    y_true = np.array(["cat", "dog", "cat"])
    y_pred = np.array(["cat", "cat", "cat"])

    expected = np.array([
        [2, 0],
        [1, 0],
    ])

    assert np.array_equal(
        confusion_matrix(y_true, y_pred, labels=["cat", "dog"]),
        expected
    )


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

    result = precision(y_true, y_pred, average="macro")
    expected = (1.0 + 0.5 + 1.0) / 3

    assert np.isclose(result, expected)


def test_recall_binary():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    assert np.isclose(recall(y_true, y_pred), 2 / 3)


def test_recall_zerodivision():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 1, 0])
    assert recall(y_true, y_pred) == 0.0


def test_recall_macro():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])

    result = recall(y_true, y_pred, average="macro")
    expected = (1.0 + 1.0 + 0.5) / 3

    assert np.isclose(result, expected)


def test_f1_binary():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    assert np.isclose(f1(y_true, y_pred), 0.8)


def test_f1_zerodivision():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])
    assert f1(y_true, y_pred) == 0.0


def test_f1_macro():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])

    result = f1(y_true, y_pred, average="macro")

    f1_class_0 = 1.0
    f1_class_1 = 2 * 0.5 * 1.0 / (0.5 + 1.0)
    f1_class_2 = 2 * 1.0 * 0.5 / (1.0 + 0.5)

    expected = (f1_class_0 + f1_class_1 + f1_class_2) / 3

    assert np.isclose(result, expected)


def test_msebasic():
    y_true = np.array([3.0, 2.5, 4.0])
    y_pred = np.array([2.5, 2.0, 4.5])
    assert mse(y_true, y_pred) == 0.25


def test_msezero():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mse(y_true, y_pred) == 0.0


def test_mse_nonnumeric():
    with pytest.raises(ValueError):
        mse(np.array(["a", "b"]), np.array(["a", "c"]))


def test_roc_curve():
    y_true = np.array([1, 0, 1, 1, 0])
    scores = np.array([0.9, 0.2, 0.4, 0.8, 0.1])

    fpr_values, tpr_values, thresholds = roc_curve(y_true, scores)

    assert fpr_values[0] == 0.0
    assert tpr_values[0] == 0.0
    assert thresholds[0] == np.inf


def test_roc_curve_oneclass_error():
    y_true = np.array([1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7])

    with pytest.raises(ValueError):
        roc_curve(y_true, scores)


def test_auc():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 1.0, 0.0])

    assert np.isclose(auc(x, y), 0.5)