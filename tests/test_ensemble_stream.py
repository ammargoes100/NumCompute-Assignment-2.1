"""
Tests for the basic tree ensemble classifier.
"""

import numpy as np
import pytest

from numcompute_stream.ensemble import EnsembleClassifier


def test_ensemble_fit_predict_simple_binary_problem():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert np.array_equal(preds, y)


def test_ensemble_predict_shape():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (4,)


def test_ensemble_handles_2d_features():
    X = np.array([
        [0.0, 1.0],
        [0.5, 1.2],
        [2.0, 3.0],
        [3.0, 3.5],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert np.array_equal(preds, y)


def test_ensemble_entropy_criterion():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(
        n_estimators=3,
        max_depth=2,
        criterion="entropy",
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)

    assert np.array_equal(preds, y)


def test_ensemble_without_bootstrap():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(
        n_estimators=3,
        max_depth=2,
        bootstrap=False,
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)

    assert np.array_equal(preds, y)


def test_ensemble_single_class():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])

    model = EnsembleClassifier(n_estimators=3, max_depth=3, random_state=42)
    model.fit(X, y)

    preds = model.predict(np.array([[10.0], [20.0]]))

    assert np.array_equal(preds, np.array([1, 1]))


def test_ensemble_predict_before_fit_raises():
    model = EnsembleClassifier()

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_ensemble_rejects_mismatched_lengths():
    X = np.array([[1.0], [2.0]])
    y = np.array([0])

    model = EnsembleClassifier()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_ensemble_rejects_nan_features():
    X = np.array([[1.0], [np.nan]])
    y = np.array([0, 1])

    model = EnsembleClassifier()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_ensemble_feature_mismatch_on_predict_raises():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    model = EnsembleClassifier()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0]]))