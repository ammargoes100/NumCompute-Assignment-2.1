"""
Tests for retained pipeline utilities.
"""

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline, FeatureUnion


class AddOne:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X + 1


class Double:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * 2


class DummyModel:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


def test_fit_transform_applies_steps_in_order():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    pipe = Pipeline([
        ("add", AddOne()),
        ("double", Double()),
    ])

    result = pipe.fit_transform(X)
    expected = (X + 1) * 2

    np.testing.assert_array_almost_equal(result, expected)


def test_fit_passes_transformed_data_to_next_step():
    X = np.array([[1.0], [2.0], [3.0]])

    pipe = Pipeline([
        ("add", AddOne()),
        ("double", Double()),
    ])

    pipe.fit(X)
    result = pipe.transform(X)
    expected = (X + 1) * 2

    np.testing.assert_array_almost_equal(result, expected)


def test_predict_raises_if_final_step_has_no_predict():
    X = np.array([[1.0], [2.0]])

    pipe = Pipeline([
        ("add", AddOne()),
        ("double", Double()),
    ])

    with pytest.raises(AttributeError):
        pipe.predict(X)


def test_predict_works_end_to_end():
    X = np.array([[1.0], [2.0], [3.0]])

    pipe = Pipeline([
        ("add", AddOne()),
        ("model", DummyModel()),
    ])

    pipe.fit(X)
    result = pipe.predict(X)

    assert result.shape == (3,)
    np.testing.assert_array_equal(result, np.zeros(3))


def test_single_step_pipeline_transformer():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    pipe = Pipeline([
        ("add", AddOne()),
    ])

    result = pipe.fit_transform(X)

    np.testing.assert_array_almost_equal(result, X + 1)


def test_feature_union_hstacks_outputs():
    X = np.array([[1.0], [2.0]])

    union = FeatureUnion([
        ("add", AddOne()),
        ("double", Double()),
    ])

    result = union.fit_transform(X)
    expected = np.hstack([X + 1, X * 2])

    np.testing.assert_array_almost_equal(result, expected)


def test_feature_union_get_transformer():
    add = AddOne()

    union = FeatureUnion([
        ("add", add),
    ])

    assert union.get_transformer("add") is add