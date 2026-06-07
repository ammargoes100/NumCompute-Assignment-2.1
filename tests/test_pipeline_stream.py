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
class StreamingScaler:
    def __init__(self):
        self.n_updates = 0
        self.mean_ = 0.0

    def partial_fit(self, X, y=None):
        self.n_updates += 1
        self.mean_ = np.mean(X)
        return self

    def transform(self, X):
        return X - self.mean_


class StreamingModel:
    def __init__(self):
        self.n_updates = 0
        self.last_shape = None

    def partial_fit(self, X, y=None):
        self.n_updates += 1
        self.last_shape = X.shape
        return self

    def predict(self, X):
        return np.ones(X.shape[0])
def test_pipeline_partial_fit_updates_transformer_and_model():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1])

    scaler = StreamingScaler()
    model = StreamingModel()

    pipe = Pipeline([
        ("scale", scaler),
        ("model", model),
    ])

    pipe.partial_fit(X, y)

    assert scaler.n_updates == 1
    assert model.n_updates == 1
    assert model.last_shape == X.shape


def test_pipeline_partial_fit_can_be_called_multiple_times():
    X1 = np.array([[1.0], [2.0]])
    y1 = np.array([0, 1])

    X2 = np.array([[3.0], [4.0]])
    y2 = np.array([1, 1])

    scaler = StreamingScaler()
    model = StreamingModel()

    pipe = Pipeline([
        ("scale", scaler),
        ("model", model),
    ])

    pipe.partial_fit(X1, y1)
    pipe.partial_fit(X2, y2)

    assert scaler.n_updates == 2
    assert model.n_updates == 2


def test_pipeline_partial_fit_predict_after_update():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1])

    pipe = Pipeline([
        ("scale", StreamingScaler()),
        ("model", StreamingModel()),
    ])

    pipe.partial_fit(X, y)
    preds = pipe.predict(X)

    np.testing.assert_array_equal(preds, np.ones(3))


def test_feature_union_partial_fit_updates_all_transformers():
    X = np.array([[1.0], [2.0]])

    first = StreamingScaler()
    second = StreamingScaler()

    union = FeatureUnion([
        ("first", first),
        ("second", second),
    ])

    union.partial_fit(X)

    assert first.n_updates == 1
    assert second.n_updates == 1