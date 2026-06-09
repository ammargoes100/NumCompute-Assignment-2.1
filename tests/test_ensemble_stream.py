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
def test_ensemble_partial_fit_first_chunk_trains_model():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.partial_fit(X, y)

    preds = model.predict(X)

    assert np.array_equal(preds, y)


def test_ensemble_partial_fit_accumulates_chunks():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])

    X2 = np.array([[2.0], [3.0]])
    y2 = np.array([1, 1])

    X_all = np.vstack([X1, X2])

    model = EnsembleClassifier(
        n_estimators=5,
        max_depth=2,
        bootstrap=False,
        random_state=42,
    )
    model.partial_fit(X1, y1)
    model.partial_fit(X2, y2)

    preds = model.predict(X_all)

    assert np.array_equal(preds, np.array([0, 0, 1, 1]))


def test_ensemble_partial_fit_updates_classes():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])

    X2 = np.array([[2.0], [3.0]])
    y2 = np.array([1, 1])

    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.partial_fit(X1, y1)

    assert np.array_equal(model.classes_, np.array([0]))

    model.partial_fit(X2, y2)

    assert np.array_equal(model.classes_, np.array([0, 1]))


def test_ensemble_partial_fit_feature_mismatch_raises():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])

    X2 = np.array([[2.0, 3.0]])
    y2 = np.array([1])

    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.partial_fit(X1, y1)

    with pytest.raises(ValueError):
        model.partial_fit(X2, y2)


def test_ensemble_partial_fit_creates_expected_number_of_estimators():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(n_estimators=7, max_depth=2, random_state=42)
    model.partial_fit(X, y)

    assert len(model.estimators_) == 7
def test_ensemble_invalid_n_estimators_raises():
    with pytest.raises(ValueError):
        EnsembleClassifier(n_estimators=0)


def test_ensemble_invalid_max_depth_raises():
    with pytest.raises(ValueError):
        EnsembleClassifier(max_depth=-1)


def test_ensemble_invalid_min_samples_split_raises():
    with pytest.raises(ValueError):
        EnsembleClassifier(min_samples_split=1)


def test_ensemble_invalid_criterion_raises():
    with pytest.raises(ValueError):
        EnsembleClassifier(criterion="gain")


def test_ensemble_invalid_max_features_raises():
    with pytest.raises(ValueError):
        EnsembleClassifier(max_features=0)


def test_ensemble_invalid_bootstrap_raises():
    with pytest.raises(ValueError):
        EnsembleClassifier(bootstrap="yes")


def test_ensemble_reset_clears_state():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    model = EnsembleClassifier(n_estimators=3, random_state=42)
    model.fit(X, y)
    model.reset()

    assert model.estimators_ == []
    assert model.classes_ is None

    with pytest.raises(ValueError):
        model.predict(X)


def test_ensemble_predict_rejects_empty_X():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    model = EnsembleClassifier()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.empty((0, 1)))


def test_ensemble_predict_rejects_nan_X():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    model = EnsembleClassifier()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[np.nan]]))


def test_ensemble_repr_contains_class_name():
    model = EnsembleClassifier(n_estimators=3)

    assert "EnsembleClassifier" in repr(model)


def test_ensemble_predictions_are_reproducible_with_seed():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model1 = EnsembleClassifier(n_estimators=5, max_depth=2, random_state=123)
    model2 = EnsembleClassifier(n_estimators=5, max_depth=2, random_state=123)

    model1.fit(X, y)
    model2.fit(X, y)

    assert np.array_equal(model1.predict(X), model2.predict(X))


def test_ensemble_max_depth_zero_predicts_majority_class():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])

    model = EnsembleClassifier(
        n_estimators=3,
        max_depth=0,
        bootstrap=False,
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(np.array([[10.0], [20.0]]))

    assert np.array_equal(preds, np.array([1, 1]))