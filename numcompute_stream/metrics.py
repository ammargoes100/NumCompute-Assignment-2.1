"""
Metric functions for NumCompute-Stream.

This module builds on the metric functions from the original NumCompute
package. The batch metrics are retained because they are still useful for
evaluating predictions on a complete dataset or on a single incoming chunk.
"""

import numpy as np


def _validate_same_shape(y_true, y_pred):
    """
    Convert inputs to arrays and check that they have the same non-empty shape.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        raise ValueError("input arrays cannot be empty")

    return y_true, y_pred


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    y_true, y_pred = _validate_same_shape(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute a confusion matrix for classification labels.

    Rows represent true labels and columns represent predicted labels.
    """
    y_true, y_pred = _validate_same_shape(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)

    if labels.size == 0:
        raise ValueError("labels cannot be empty")

    label_to_index = {label: index for index, label in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label not in label_to_index or pred_label not in label_to_index:
            raise ValueError("unknown label found")

        row = label_to_index[true_label]
        col = label_to_index[pred_label]
        cm[row, col] += 1

    return cm


def precision(y_true, y_pred, average="binary", positive_label=1):
    """
    Compute binary or macro-averaged precision.
    """
    y_true, y_pred = _validate_same_shape(y_true, y_pred)

    if average == "binary":
        tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
        fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
        return 0.0 if (tp + fp) == 0 else float(tp / (tp + fp))

    if average == "macro":
        labels = np.unique(np.concatenate((y_true, y_pred)))
        scores = []

        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            score = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
            scores.append(score)

        return float(np.mean(scores))

    raise ValueError("average must be 'binary' or 'macro'")


def recall(y_true, y_pred, average="binary", positive_label=1):
    """
    Compute binary or macro-averaged recall.
    """
    y_true, y_pred = _validate_same_shape(y_true, y_pred)

    if average == "binary":
        tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
        fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
        return 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))

    if average == "macro":
        labels = np.unique(np.concatenate((y_true, y_pred)))
        scores = []

        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            score = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
            scores.append(score)

        return float(np.mean(scores))

    raise ValueError("average must be 'binary' or 'macro'")


def f1(y_true, y_pred, average="binary", positive_label=1):
    """
    Compute binary or macro-averaged F1 score.
    """
    y_true, y_pred = _validate_same_shape(y_true, y_pred)

    if average == "binary":
        p = precision(y_true, y_pred, average="binary", positive_label=positive_label)
        r = recall(y_true, y_pred, average="binary", positive_label=positive_label)
        return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))

    if average == "macro":
        labels = np.unique(np.concatenate((y_true, y_pred)))
        scores = []

        for label in labels:
            p = precision(y_true, y_pred, average="binary", positive_label=label)
            r = recall(y_true, y_pred, average="binary", positive_label=label)
            score = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
            scores.append(score)

        return float(np.mean(scores))

    raise ValueError("average must be 'binary' or 'macro'")


def mse(y_true, y_pred):
    """
    Compute mean squared error.
    """
    y_true, y_pred = _validate_same_shape(y_true, y_pred)

    if not np.issubdtype(y_true.dtype, np.number):
        raise ValueError("y_true must be numeric")

    if not np.issubdtype(y_pred.dtype, np.number):
        raise ValueError("y_pred must be numeric")

    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true, y_score, positive_label=1):
    """
    Compute false positive rate, true positive rate, and thresholds.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    if y_true.size == 0:
        raise ValueError("input arrays cannot be empty")

    y_binary = (y_true == positive_label).astype(int)

    pos = np.sum(y_binary == 1)
    neg = np.sum(y_binary == 0)

    if pos == 0 or neg == 0:
        raise ValueError("both positive and negative classes are needed")

    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1]]

    tpr = []
    fpr = []

    for threshold in thresholds:
        pred_positive = y_score >= threshold
        tp = np.sum((y_binary == 1) & pred_positive)
        fp = np.sum((y_binary == 0) & pred_positive)

        tpr.append(tp / pos)
        fpr.append(fp / neg)

    return np.asarray(fpr), np.asarray(tpr), thresholds


def auc(x, y):
    """
    Compute area under a curve using the trapezoidal rule.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.ndim != 1:
        raise ValueError("x and y must be 1D arrays")

    if x.size < 2:
        raise ValueError("at least two points are required")

    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))
class StreamingAccuracy:
    """
    Accumulate classification accuracy over multiple chunks.
    """

    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_same_shape(y_true, y_pred)

        self.correct_ += int(np.sum(y_true == y_pred))
        self.total_ += int(y_true.size)
        return self

    def result(self):
        if self.total_ == 0:
            return 0.0

        return float(self.correct_ / self.total_)

    def reset(self):
        self.correct_ = 0
        self.total_ = 0
        return self


class StreamingConfusionMatrix:
    """
    Accumulate a confusion matrix over multiple chunks.
    """

    def __init__(self, labels):
        labels = np.asarray(labels)

        if labels.size == 0:
            raise ValueError("labels cannot be empty")

        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.reset()

    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_same_shape(y_true, y_pred)

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label not in self.label_to_index or pred_label not in self.label_to_index:
                raise ValueError("unknown label found")

            row = self.label_to_index[true_label]
            col = self.label_to_index[pred_label]
            self.matrix_[row, col] += 1

        return self

    def result(self):
        return self.matrix_.copy()

    def reset(self):
        self.matrix_ = np.zeros((self.labels.size, self.labels.size), dtype=int)
        return self


class StreamingPrecision:
    """
    Accumulate precision over multiple classification chunks.
    """

    def __init__(self, average="binary", positive_label=1, labels=None):
        self.average = average
        self.positive_label = positive_label
        self.labels = labels
        self.y_true_parts_ = []
        self.y_pred_parts_ = []

    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_same_shape(y_true, y_pred)

        self.y_true_parts_.append(y_true.copy())
        self.y_pred_parts_.append(y_pred.copy())
        return self

    def result(self):
        if len(self.y_true_parts_) == 0:
            return 0.0

        y_true = np.concatenate(self.y_true_parts_)
        y_pred = np.concatenate(self.y_pred_parts_)

        return precision(
            y_true,
            y_pred,
            average=self.average,
            positive_label=self.positive_label,
        )

    def reset(self):
        self.y_true_parts_ = []
        self.y_pred_parts_ = []
        return self


class StreamingRecall:
    """
    Accumulate recall over multiple classification chunks.
    """

    def __init__(self, average="binary", positive_label=1):
        self.average = average
        self.positive_label = positive_label
        self.y_true_parts_ = []
        self.y_pred_parts_ = []

    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_same_shape(y_true, y_pred)

        self.y_true_parts_.append(y_true.copy())
        self.y_pred_parts_.append(y_pred.copy())
        return self

    def result(self):
        if len(self.y_true_parts_) == 0:
            return 0.0

        y_true = np.concatenate(self.y_true_parts_)
        y_pred = np.concatenate(self.y_pred_parts_)

        return recall(
            y_true,
            y_pred,
            average=self.average,
            positive_label=self.positive_label,
        )

    def reset(self):
        self.y_true_parts_ = []
        self.y_pred_parts_ = []
        return self


class StreamingF1:
    """
    Accumulate F1 score over multiple classification chunks.
    """

    def __init__(self, average="binary", positive_label=1):
        self.average = average
        self.positive_label = positive_label
        self.y_true_parts_ = []
        self.y_pred_parts_ = []

    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_same_shape(y_true, y_pred)

        self.y_true_parts_.append(y_true.copy())
        self.y_pred_parts_.append(y_pred.copy())
        return self

    def result(self):
        if len(self.y_true_parts_) == 0:
            return 0.0

        y_true = np.concatenate(self.y_true_parts_)
        y_pred = np.concatenate(self.y_pred_parts_)

        return f1(
            y_true,
            y_pred,
            average=self.average,
            positive_label=self.positive_label,
        )

    def reset(self):
        self.y_true_parts_ = []
        self.y_pred_parts_ = []
        return self


class StreamingMSE:
    """
    Accumulate mean squared error over multiple chunks.
    """

    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_same_shape(y_true, y_pred)

        if not np.issubdtype(y_true.dtype, np.number):
            raise ValueError("y_true must be numeric")

        if not np.issubdtype(y_pred.dtype, np.number):
            raise ValueError("y_pred must be numeric")

        errors = (y_true - y_pred) ** 2
        self.sum_squared_error_ += float(np.sum(errors))
        self.total_ += int(y_true.size)
        return self

    def result(self):
        if self.total_ == 0:
            return 0.0

        return float(self.sum_squared_error_ / self.total_)

    def reset(self):
        self.sum_squared_error_ = 0.0
        self.total_ = 0
        return self