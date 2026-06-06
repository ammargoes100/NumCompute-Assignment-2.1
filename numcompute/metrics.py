"""
Module: metrics.py
Description:Here,metrics.py is providing Numpy based implementations of classification and
regression such as accuracy,precision, F1 score, confusion metrix, mean squared error,
recall,AUC and ROC curve.   
"""





import numpy as np


def _validate_same_shape(b_true, b_pred):
    b_true = np.asarray(b_true)
    b_pred = np.asarray(b_pred)

    if b_true.shape != b_pred.shape:
        raise ValueError("b_true and b_pred should have same shape.")

    if b_true.size == 0:
        raise ValueError("It is not possible for input arrays to be empty.")

    return b_true, b_pred

#accuracy is being computed
def accuracy(b_true, b_pred):
    b_true, b_pred = _validate_same_shape(b_true, b_pred)
    return float(np.mean(b_true == b_pred))

#confusion matrix
def confusion_matrix(b_true, b_pred, labels=None):
    b_true, b_pred = _validate_same_shape(b_true, b_pred)

    if labels is None:
        labels = np.unique(np.concatenate((b_true, b_pred)))
    else:
        labels = np.asarray(labels)

    if labels.size == 0:
        raise ValueError("It is not possible for labels to be empty.")

    label_to_index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=int)

    for t, p in zip(b_true, b_pred):
        if t not in label_to_index or p not in label_to_index:
            raise ValueError("Unknown label found.")
        cm[label_to_index[t], label_to_index[p]] += 1

    return cm

#precision being calculated
def precision(b_true, b_pred, average="binary", positive_label=1):
    b_true, b_pred = _validate_same_shape(b_true, b_pred)

    if average == "binary":
        tp = np.sum((b_true == positive_label) & (b_pred == positive_label))
        fp = np.sum((b_true != positive_label) & (b_pred == positive_label))
        return 0.0 if (tp + fp) == 0 else float(tp / (tp + fp))

    elif average == "macro":
        labels = np.unique(np.concatenate((b_true, b_pred)))
        scores = []

        for label in labels:
            tp = np.sum((b_true == label) & (b_pred == label))
            fp = np.sum((b_true != label) & (b_pred == label))
            scores.append(0.0 if (tp + fp) == 0 else tp / (tp + fp))

        return float(np.mean(scores))

    else:
        raise ValueError("average should be 'macro' or 'binary'")

#recall
def recall(b_true, b_pred, average="binary", positive_label=1):
    
    b_true, b_pred = _validate_same_shape(b_true, b_pred)

    if average == "binary":
        tp = np.sum((b_true == positive_label) & (b_pred == positive_label))
        fn = np.sum((b_true == positive_label) & (b_pred != positive_label))
        return 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))

    elif average == "macro":
        labels = np.unique(np.concatenate((b_true, b_pred)))
        scores = []

        for label in labels:
            tp = np.sum((b_true == label) & (b_pred == label))
            fn = np.sum((b_true == label) & (b_pred != label))
            scores.append(0.0 if (tp + fn) == 0 else tp / (tp + fn))

        return float(np.mean(scores))

    else:
        raise ValueError("average must be 'binary' or 'macro'")

#f1 score
def f1(b_true, b_pred, average="binary", positive_label=1):
    b_true, b_pred = _validate_same_shape(b_true, b_pred)

    if average == "binary":
        p = precision(b_true, b_pred, "binary", positive_label)
        r = recall(b_true, b_pred, "binary", positive_label)
        return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))

    elif average == "macro":
        labels = np.unique(np.concatenate((b_true, b_pred)))
        scores = []

        for label in labels:
            tp = np.sum((b_true == label) & (b_pred == label))
            fp = np.sum((b_true != label) & (b_pred == label))
            fn = np.sum((b_true == label) & (b_pred != label))

            p = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
            r = 0.0 if (tp + fn) == 0 else tp / (tp + fn)

            scores.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))

        return float(np.mean(scores))

    else:
        raise ValueError("average must be 'binary' or 'macro'")


def mse(b_true, b_pred):
    b_true, b_pred = _validate_same_shape(b_true, b_pred)

    if not np.issubdtype(b_true.dtype, np.number):
        raise ValueError("b_true should be numeric.")
    if not np.issubdtype(b_pred.dtype, np.number):
        raise ValueError("b_pred must be numeric.")

    return float(np.mean((b_true - b_pred) ** 2))

#roc curve
def roc_curve(b_true, b_score, positive_label=1):
    b_true = np.asarray(b_true)
    b_score = np.asarray(b_score)

    if b_true.shape != b_score.shape:
        raise ValueError("Shapes must match.")
    if b_true.size == 0:
        raise ValueError("Empty input.")

    y_bin = (b_true == positive_label).astype(int)

    pos = np.sum(y_bin == 1)
    neg = np.sum(y_bin == 0)

    if pos == 0 or neg == 0:
        raise ValueError("both classes are needed.")

    thresholds = np.r_[np.inf, np.sort(np.unique(b_score))[::-1]]

    tpr = []
    fpr = []

    for t in thresholds:
        pred = b_score >= t
        tp = np.sum((y_bin == 1) & pred)
        fp = np.sum((y_bin == 0) & pred)

        tpr.append(tp / pos)
        fpr.append(fp / neg)

    return np.array(fpr), np.array(tpr), thresholds
#auc being computed

def auc(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must match.")
    if x.ndim != 1:
        raise ValueError("1D arrays required.")
    if x.size < 2:
        raise ValueError("Need at least two points.")

    order = np.argsort(x)

    
    return float(np.trapezoid(y[order], x[order]))