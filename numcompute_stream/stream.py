"""
Streaming training utilities for NumCompute-Stream.

This module provides StreamTrainer, a lightweight controller for training a
model or pipeline on incoming data chunks.
"""

import numpy as np


class StreamTrainer:
    """
    Manage chunk-wise training and evaluation for streaming models.

    Parameters
    ----------
    model : object
        Model or pipeline implementing partial_fit() and predict().
    metric_fn : callable or None, default=None
        Optional metric function taking (y_true, y_pred). If None, accuracy is used.
    """

    def __init__(self, model, metric_fn=None):
        self.model = model
        self.metric_fn = metric_fn if metric_fn is not None else self._accuracy
        self.history_ = []

    def fit_chunk(self, X, y):
        """
        Fit the model on one incoming chunk.
        """
        X, y = self._validate_X_y(X, y)

        if not hasattr(self.model, "partial_fit"):
            raise AttributeError("model must implement partial_fit()")

        self.model.partial_fit(X, y)

        return self

    def score_chunk(self, X, y):
        """
        Predict and score one chunk.
        """
        X, y = self._validate_X_y(X, y)

        if not hasattr(self.model, "predict"):
            raise AttributeError("model must implement predict()")

        y_pred = self.model.predict(X)
        score = self.metric_fn(y, y_pred)

        return score

    def fit_stream(self, chunks):
        """
        Fit the model on an iterable of (X_chunk, y_chunk) pairs.
        """
        for X_chunk, y_chunk in chunks:
            self.fit_chunk(X_chunk, y_chunk)

        return self

    def _validate_X_y(self, X, y):
        """
        Validate chunk data.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        if np.isnan(X).any():
            raise ValueError("X must not contain NaN values")

        return X, y

    def _accuracy(self, y_true, y_pred):
        """
        Compute simple classification accuracy.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        return float(np.mean(y_true == y_pred))