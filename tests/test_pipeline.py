"""
Tests for pipeline.py
"""
import numpy as np
import pytest
from numcompute.pipeline import Pipeline


# dummy transformer that adds 1 to every value
class AddOne:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X + 1


# dummy transformer that doubles every value
class Double:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X * 2


# dummy estimator with predict
class DummyModel:
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros(X.shape[0])


def test_fit_transform_applies_steps_in_order():
    # AddOne then Double: (X + 1) * 2
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipe = Pipeline([('add', AddOne()), ('double', Double())])
    result = pipe.fit_transform(X)
    expected = (X + 1) * 2
    np.testing.assert_array_almost_equal(result, expected)


def test_fit_passes_transformed_data_to_next_step():
    # fit should pass transformed X between steps not raw X
    X = np.array([[1.0], [2.0], [3.0]])
    pipe = Pipeline([('add', AddOne()), ('double', Double())])
    pipe.fit(X)
    result = pipe.transform(X)
    expected = (X + 1) * 2
    np.testing.assert_array_almost_equal(result, expected)


def test_predict_raises_if_final_step_has_no_predict():
    # last step must have predict() or we raise AttributeError
    X = np.array([[1.0], [2.0]])
    pipe = Pipeline([('add', AddOne()), ('double', Double())])
    with pytest.raises(AttributeError):
        pipe.predict(X)
        
def test_predict_works_end_to_end():
    X = np.array([[1.0], [2.0], [3.0]])
    pipe = Pipeline([('add', AddOne()), ('model', DummyModel())])
    pipe.fit(X)
    result = pipe.predict(X)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, np.zeros(3))


def test_single_step_pipeline():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipe = Pipeline([('add', AddOne())])
    result = pipe.fit_transform(X)
    np.testing.assert_array_almost_equal(result, X + 1)