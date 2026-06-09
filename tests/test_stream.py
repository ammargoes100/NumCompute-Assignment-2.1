"""
Tests for the stream trainer.
"""

import numpy as np
import pytest

from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


class DummyStreamingModel:
    def __init__(self):
        self.fit_calls = 0
        self.last_X_shape = None
        self.majority_class_ = None

    def partial_fit(self, X, y):
        self.fit_calls += 1
        self.last_X_shape = X.shape
        values, counts = np.unique(y, return_counts=True)
        self.majority_class_ = values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.majority_class_)


class NoPartialFitModel:
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


class NoPredictModel:
    def partial_fit(self, X, y):
        return self


def test_stream_trainer_fit_chunk_calls_partial_fit():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    trainer.fit_chunk(X, y)

    assert model.fit_calls == 1
    assert model.last_X_shape == (3, 1)


def test_stream_trainer_score_chunk_returns_accuracy():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    trainer.fit_chunk(X, y)
    score = trainer.score_chunk(X, y)

    assert 0.0 <= score <= 1.0


def test_stream_trainer_fit_stream_multiple_chunks():
    chunks = [
        (np.array([[0.0], [1.0]]), np.array([0, 0])),
        (np.array([[2.0], [3.0]]), np.array([1, 1])),
    ]

    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    trainer.fit_stream(chunks)

    assert model.fit_calls == 2


def test_stream_trainer_works_with_decision_tree():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])
    X2 = np.array([[2.0], [3.0]])
    y2 = np.array([1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    trainer = StreamTrainer(tree)

    trainer.fit_chunk(X1, y1)
    trainer.fit_chunk(X2, y2)

    X_all = np.vstack([X1, X2])
    y_all = np.concatenate([y1, y2])

    score = trainer.score_chunk(X_all, y_all)

    assert score == 1.0


def test_stream_trainer_rejects_model_without_partial_fit():
    trainer = StreamTrainer(NoPartialFitModel())

    with pytest.raises(AttributeError):
        trainer.fit_chunk(np.array([[1.0]]), np.array([0]))


def test_stream_trainer_rejects_model_without_predict():
    trainer = StreamTrainer(NoPredictModel())

    with pytest.raises(AttributeError):
        trainer.score_chunk(np.array([[1.0]]), np.array([0]))


def test_stream_trainer_rejects_mismatched_lengths():
    trainer = StreamTrainer(DummyStreamingModel())

    with pytest.raises(ValueError):
        trainer.fit_chunk(np.array([[1.0], [2.0]]), np.array([0]))


def test_stream_trainer_rejects_empty_X():
    trainer = StreamTrainer(DummyStreamingModel())

    with pytest.raises(ValueError):
        trainer.fit_chunk(np.empty((0, 2)), np.array([]))


def test_stream_trainer_rejects_nan_X():
    trainer = StreamTrainer(DummyStreamingModel())

    with pytest.raises(ValueError):
        trainer.fit_chunk(np.array([[1.0], [np.nan]]), np.array([0, 1]))