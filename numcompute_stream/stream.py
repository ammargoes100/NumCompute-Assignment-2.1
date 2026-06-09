"""
Streaming training utilities for NumCompute-Stream.

This module provides StreamTrainer, a lightweight controller for training a
model or pipeline on incoming data chunks.

It records per-chunk scores, cumulative accuracy, fit time, prediction time,
and an approximate memory footprint so the logs can be used by benchmarking
scripts and visualisation functions.
"""

import sys
import time
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
        if model is None:
            raise ValueError("model cannot be None")

        if metric_fn is not None and not callable(metric_fn):
            raise ValueError("metric_fn must be callable")

        self.model = model
        self.metric_fn = metric_fn if metric_fn is not None else self._accuracy
        self.history_ = []
        self.n_chunks_seen_ = 0
        self.n_samples_seen_ = 0
        self.correct_seen_ = 0

    def fit_chunk(self, X, y):
        """
        Fit the model on one incoming chunk.
        """
        X, y = self._validate_X_y(X, y)

        if not hasattr(self.model, "partial_fit"):
            raise AttributeError("model must implement partial_fit()")

        start_time = time.perf_counter()
        self.model.partial_fit(X, y)
        fit_time = time.perf_counter() - start_time

        return {
            "fit_time": fit_time,
            "n_samples": int(X.shape[0]),
        }

    def score_chunk(self, X, y):
        """
        Predict and score one chunk.
        """
        X, y = self._validate_X_y(X, y)

        if not hasattr(self.model, "predict"):
            raise AttributeError("model must implement predict()")

        start_time = time.perf_counter()
        y_pred = self.model.predict(X)
        predict_time = time.perf_counter() - start_time

        y_pred = np.asarray(y_pred)

        if y_pred.shape != y.shape:
            raise ValueError("model predictions must have the same shape as y")

        score = self.metric_fn(y, y_pred)

        return {
            "score": float(score),
            "predict_time": predict_time,
            "y_pred": y_pred,
        }

    def fit_score_chunk(self, X, y):
        """
        Fit on one chunk, then score that same chunk and log the result.
        """
        X, y = self._validate_X_y(X, y)

        fit_info = self.fit_chunk(X, y)
        score_info = self.score_chunk(X, y)

        correct = int(np.sum(score_info["y_pred"] == y))
        self.correct_seen_ += correct
        self.n_samples_seen_ += int(y.shape[0])
        self.n_chunks_seen_ += 1

        cumulative_accuracy = self.correct_seen_ / self.n_samples_seen_

        log_entry = {
            "chunk_index": self.n_chunks_seen_,
            "n_samples": int(y.shape[0]),
            "score": score_info["score"],
            "cumulative_accuracy": float(cumulative_accuracy),
            "fit_time": fit_info["fit_time"],
            "predict_time": score_info["predict_time"],
            "memory_bytes": self._estimate_memory_bytes(),
        }

        self.history_.append(log_entry)

        return log_entry

    def fit_stream(self, chunks):
        """
        Fit and score the model on an iterable of (X_chunk, y_chunk) pairs.
        """
        for chunk in chunks:
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise ValueError("each stream chunk must be a tuple (X_chunk, y_chunk)")

            X_chunk, y_chunk = chunk
            self.fit_score_chunk(X_chunk, y_chunk)

        return self

    def get_history(self, key=None):
        """
        Return full history or one logged value across chunks.
        """
        if key is None:
            return list(self.history_)

        if len(self.history_) == 0:
            return []

        if key not in self.history_[0]:
            raise KeyError(f"history key '{key}' not found")

        return [entry[key] for entry in self.history_]

    def reset_logs(self):
        """
        Reset stream logs and cumulative counters.
        """
        self.history_ = []
        self.n_chunks_seen_ = 0
        self.n_samples_seen_ = 0
        self.correct_seen_ = 0

        return self

    def _estimate_memory_bytes(self):
        """
        Estimate memory usage of the model object.

        This is a lightweight approximation using sys.getsizeof().
        """
        total = sys.getsizeof(self.model)

        for value in getattr(self.model, "__dict__", {}).values():
            total += sys.getsizeof(value)

            if isinstance(value, np.ndarray):
                total += value.nbytes

        return int(total)

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

    def __repr__(self):
        return (
            "StreamTrainer("
            f"model={self.model.__class__.__name__}, "
            f"n_chunks_seen_={self.n_chunks_seen_}, "
            f"n_samples_seen_={self.n_samples_seen_})"
        )