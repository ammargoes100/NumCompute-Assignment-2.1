"""
Tests for the basic decision tree classifier.
"""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


def test_tree_fit_predict_simple_binary_problem():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, y)


def test_tree_predict_shape():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert preds.shape == (4,)


def test_tree_handles_2d_features():
    X = np.array([
        [0.0, 1.0],
        [0.5, 1.2],
        [2.0, 3.0],
        [3.0, 3.5],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, y)


def test_tree_entropy_criterion():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2, criterion="entropy")
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, y)


def test_tree_single_class_becomes_leaf():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    preds = tree.predict(np.array([[10.0], [20.0]]))

    assert np.array_equal(preds, np.array([1, 1]))


def test_tree_predict_before_fit_raises():
    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.predict(np.array([[1.0]]))


def test_tree_rejects_mismatched_lengths():
    X = np.array([[1.0], [2.0]])
    y = np.array([0])

    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.fit(X, y)


def test_tree_rejects_nan_features():
    X = np.array([[1.0], [np.nan]])
    y = np.array([0, 1])

    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.fit(X, y)


def test_tree_rejects_empty_training_data():
    X = np.empty((0, 2))
    y = np.array([])

    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.fit(X, y)


def test_tree_feature_mismatch_on_predict_raises():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    with pytest.raises(ValueError):
        tree.predict(np.array([[1.0, 2.0]]))
def test_tree_partial_fit_first_chunk_trains_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.partial_fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, y)


def test_tree_partial_fit_accumulates_chunks():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])

    X2 = np.array([[2.0], [3.0]])
    y2 = np.array([1, 1])

    X_all = np.vstack([X1, X2])
    y_all = np.concatenate([y1, y2])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.partial_fit(X1, y1)
    tree.partial_fit(X2, y2)

    preds = tree.predict(X_all)

    assert np.array_equal(preds, y_all)


def test_tree_partial_fit_updates_classes():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])

    X2 = np.array([[2.0], [3.0]])
    y2 = np.array([1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.partial_fit(X1, y1)

    assert np.array_equal(tree.classes_, np.array([0]))

    tree.partial_fit(X2, y2)

    assert np.array_equal(tree.classes_, np.array([0, 1]))


def test_tree_partial_fit_feature_mismatch_raises():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])

    X2 = np.array([[2.0, 3.0]])
    y2 = np.array([1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.partial_fit(X1, y1)

    with pytest.raises(ValueError):
        tree.partial_fit(X2, y2)
def test_tree_invalid_max_depth_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=-1)


def test_tree_invalid_min_samples_split_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1)


def test_tree_invalid_criterion_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="gain")


def test_tree_invalid_max_features_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_features=0)


def test_tree_max_depth_zero_predicts_majority_class():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=0)
    tree.fit(X, y)

    preds = tree.predict(np.array([[10.0], [20.0]]))

    assert np.array_equal(preds, np.array([1, 1]))


def test_tree_min_samples_split_stops_growth():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=10)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, np.array([0, 0, 0, 0]))


def test_tree_reset_clears_state():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    tree.reset()

    assert tree.tree_ is None
    assert tree.classes_ is None

    with pytest.raises(ValueError):
        tree.predict(X)


def test_tree_predict_rejects_empty_X():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    with pytest.raises(ValueError):
        tree.predict(np.empty((0, 1)))


def test_tree_predict_rejects_nan_X():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    with pytest.raises(ValueError):
        tree.predict(np.array([[np.nan]]))


def test_tree_repr_contains_class_name():
    tree = DecisionTreeClassifier(max_depth=2)

    assert "DecisionTreeClassifier" in repr(tree)