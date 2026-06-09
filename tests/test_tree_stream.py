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