"""
Tests for retained preprocessing transformers.
"""

import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
)


class TestStandardScaler:
    def test_fit_transform_1d(self):
        X = np.array([1, 2, 3, 4, 5])
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        assert np.isclose(X_scaled.mean(), 0)
        assert np.isclose(X_scaled.std(ddof=0), 1)

    def test_fit_transform_2d(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled.mean(axis=0), [0, 0])
        assert np.allclose(X_scaled.std(axis=0, ddof=0), [1, 1])

    def test_with_mean_false(self):
        X = np.array([[1, 2], [3, 4]])
        scaler = StandardScaler(with_mean=False, with_std=True)

        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled.mean(axis=0), [2, 3])
        assert np.allclose(X_scaled.std(axis=0, ddof=0), [1, 1])

    def test_with_std_false(self):
        X = np.array([[1, 2], [3, 4]])
        scaler = StandardScaler(with_mean=True, with_std=False)

        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled.mean(axis=0), [0, 0])
        assert np.allclose(X_scaled.std(axis=0, ddof=0), [1, 1])

    def test_constant_feature(self):
        X = np.array([[1, 2], [1, 4], [1, 6]])
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled[:, 0], 0)
        assert np.isclose(X_scaled[:, 1].std(ddof=0), 1)

    def test_empty_input_raises(self):
        scaler = StandardScaler()

        with pytest.raises(ValueError, match="0 samples"):
            scaler.fit(np.empty((0, 3)))

    def test_feature_mismatch_raises(self):
        scaler = StandardScaler().fit(np.array([[1, 2], [3, 4]]))

        with pytest.raises(ValueError, match="Expected 2 features"):
            scaler.transform(np.array([[1, 2, 3]]))

    def test_non_contiguous_input(self):
        X = np.arange(12).reshape(6, 2)[::2]
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-12)
        assert np.allclose(X_scaled.std(axis=0, ddof=0), 1, atol=1e-12)

    def test_3d_input_raises(self):
        X = np.ones((5, 2, 3))
        scaler = StandardScaler()

        with pytest.raises(ValueError):
            scaler.fit(X)

    def test_transform_empty_after_fit(self):
        scaler = StandardScaler().fit(np.array([[1, 2], [3, 4]]))
        X_empty = np.empty((0, 2))

        X_scaled = scaler.transform(X_empty)

        assert X_scaled.shape == (0, 2)

    def test_large_numbers(self):
        X = np.array([[1e10, 2e10], [3e10, 4e10]])
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        assert np.isfinite(X_scaled).all()


class TestMinMaxScaler:
    def test_default_range(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()

        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled.min(axis=0), [0, 0])
        assert np.allclose(X_scaled.max(axis=0), [1, 1])

    def test_custom_range(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler(feature_range=(-1, 1))

        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled.min(axis=0), [-1, -1])
        assert np.allclose(X_scaled.max(axis=0), [1, 1])

    def test_constant_feature_uses_range_min(self):
        X = np.array([[2, 1], [2, 3], [2, 5]])
        scaler = MinMaxScaler(feature_range=(0, 1))

        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled[:, 0], 0)

    def test_1d_input(self):
        X = np.array([1, 2, 3, 4, 5])
        scaler = MinMaxScaler()

        X_scaled = scaler.fit_transform(X)

        assert X_scaled.min() == 0
        assert X_scaled.max() == 1

    def test_empty_input_raises(self):
        scaler = MinMaxScaler()

        with pytest.raises(ValueError):
            scaler.fit(np.empty((0, 2)))

    def test_invalid_feature_range_raises(self):
        with pytest.raises(ValueError, match="feature_range must be a tuple"):
            MinMaxScaler(feature_range=(5, 0))

    def test_integer_input_becomes_float(self):
        X = np.array([[1, 2], [3, 4]], dtype=int)
        scaler = MinMaxScaler()

        X_scaled = scaler.fit_transform(X)

        assert X_scaled.dtype == np.float64


class TestOneHotEncoder:
    def test_single_feature(self):
        X = np.array([1, 2, 1, 3])
        enc = OneHotEncoder()

        X_enc = enc.fit_transform(X)

        assert len(enc.categories_[0]) == 3
        assert X_enc.shape == (4, 3)
        assert (X_enc.sum(axis=1) == 1).all()

    def test_two_features(self):
        X = np.array([[1, "a"], [2, "b"], [1, "b"]], dtype=object)
        enc = OneHotEncoder()

        X_enc = enc.fit_transform(X)

        assert X_enc.shape == (3, 4)
        assert (X_enc.sum(axis=1) == 2).all()

    def test_unknown_category_error(self):
        X = np.array([1, 2, 3])
        enc = OneHotEncoder(handle_unknown="error")
        enc.fit(X)

        with pytest.raises(ValueError, match="Unknown category '4'"):
            enc.transform(np.array([4]))

    def test_unknown_category_ignore(self):
        X = np.array([1, 2, 3])
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(X)

        X_enc = enc.transform(np.array([4, 1]))

        assert (X_enc[0] == 0).all()
        assert X_enc[1].sum() == 1

    def test_empty_fit_raises(self):
        enc = OneHotEncoder()

        with pytest.raises(ValueError):
            enc.fit(np.empty((0, 2)))

    def test_single_category(self):
        X = np.array([[1], [1], [1]])
        enc = OneHotEncoder()

        X_enc = enc.fit_transform(X)

        assert X_enc.shape == (3, 1)
        assert np.all(X_enc == 1)

    def test_nan_input_raises(self):
        enc = OneHotEncoder()

        with pytest.raises(ValueError, match="NaN"):
            enc.fit(np.array([[1, np.nan]]))


class TestSimpleImputer:
    def test_constant_strategy(self):
        X = np.array([[1, np.nan], [3, 4], [np.nan, 6]])
        imputer = SimpleImputer(strategy="constant", fill_value=0)

        X_imp = imputer.fit_transform(X)
        expected = np.array([[1, 0], [3, 4], [0, 6]])

        assert np.array_equal(X_imp, expected)

    def test_mean_strategy(self):
        X = np.array([[1, np.nan], [3, 4], [5, 6]])
        imputer = SimpleImputer(strategy="mean")

        X_imp = imputer.fit_transform(X)
        expected = np.array([[1, 5], [3, 4], [5, 6]])

        assert np.array_equal(X_imp, expected)

    def test_median_strategy(self):
        X = np.array([[1, np.nan], [3, 4], [5, 6]])
        imputer = SimpleImputer(strategy="median")

        X_imp = imputer.fit_transform(X)
        expected = np.array([[1, 5], [3, 4], [5, 6]])

        assert np.array_equal(X_imp, expected)

    def test_all_nan_column(self):
        X = np.array([[1, np.nan], [2, np.nan], [3, np.nan]])
        imputer = SimpleImputer(strategy="mean")

        X_imp = imputer.fit_transform(X)

        assert np.all(X_imp[:, 1] == 0)

    def test_1d_input(self):
        X = np.array([1, np.nan, 3])
        imputer = SimpleImputer(strategy="constant", fill_value=0)

        X_imp = imputer.fit_transform(X)

        assert np.array_equal(X_imp, [[1], [0], [3]])

    def test_empty_fit_raises(self):
        imputer = SimpleImputer()

        with pytest.raises(ValueError):
            imputer.fit(np.empty((0, 2)))

    def test_feature_mismatch_raises(self):
        imputer = SimpleImputer().fit(np.array([[1, 2], [3, 4]]))

        with pytest.raises(ValueError):
            imputer.transform(np.array([[1, 2, 3]]))

    def test_transform_no_nan(self):
        X = np.array([[1, 2], [3, 4]])
        imputer = SimpleImputer(strategy="mean").fit(X)

        X_imp = imputer.transform(X)

        assert np.array_equal(X_imp, X)
class TestStreamingPreprocessing:
    def test_standard_scaler_partial_fit_matches_full_fit(self):
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        X_all = np.vstack([X1, X2])

        stream_scaler = StandardScaler()
        stream_scaler.partial_fit(X1)
        stream_scaler.partial_fit(X2)

        batch_scaler = StandardScaler().fit(X_all)

        assert np.allclose(stream_scaler.mean_, batch_scaler.mean_)
        assert np.allclose(stream_scaler.scale_, batch_scaler.scale_)
        assert stream_scaler.n_samples_seen_ == 4

    def test_minmax_scaler_partial_fit_updates_range(self):
        X1 = np.array([[2.0, 10.0], [4.0, 20.0]])
        X2 = np.array([[0.0, 30.0], [8.0, 5.0]])

        scaler = MinMaxScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)

        assert np.allclose(scaler.data_min_, [0.0, 5.0])
        assert np.allclose(scaler.data_max_, [8.0, 30.0])
        assert scaler.n_samples_seen_ == 4

    def test_one_hot_encoder_partial_fit_adds_new_categories(self):
        enc = OneHotEncoder()

        enc.partial_fit(np.array(["red", "blue"]))
        enc.partial_fit(np.array(["green"]))

        assert set(enc.categories_[0].tolist()) == {"red", "blue", "green"}

        X_enc = enc.transform(np.array(["green", "red"]))

        assert X_enc.shape == (2, 3)
        assert (X_enc.sum(axis=1) == 1).all()

    def test_simple_imputer_mean_partial_fit_updates_statistics(self):
        X1 = np.array([[1.0, np.nan], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0]])

        imputer = SimpleImputer(strategy="mean")
        imputer.partial_fit(X1)
        imputer.partial_fit(X2)

        assert np.allclose(imputer.statistics_, [3.0, 5.0])
        assert imputer.n_samples_seen_ == 3

    def test_simple_imputer_median_partial_fit_updates_statistics(self):
        X1 = np.array([[1.0, np.nan], [5.0, 4.0]])
        X2 = np.array([[9.0, 8.0]])

        imputer = SimpleImputer(strategy="median")
        imputer.partial_fit(X1)
        imputer.partial_fit(X2)

        assert np.allclose(imputer.statistics_, [5.0, 6.0])

    def test_partial_fit_feature_mismatch_raises(self):
        scaler = StandardScaler()
        scaler.partial_fit(np.array([[1.0, 2.0]]))

        with pytest.raises(ValueError):
            scaler.partial_fit(np.array([[1.0, 2.0, 3.0]]))
class TestPreprocessingValidation:
    def test_standard_scaler_transform_before_fit_raises(self):
        scaler = StandardScaler()

        with pytest.raises(ValueError):
            scaler.transform(np.array([[1.0, 2.0]]))

    def test_minmax_scaler_transform_before_fit_raises(self):
        scaler = MinMaxScaler()

        with pytest.raises(ValueError):
            scaler.transform(np.array([[1.0, 2.0]]))

    def test_one_hot_transform_before_fit_raises(self):
        enc = OneHotEncoder()

        with pytest.raises(ValueError):
            enc.transform(np.array(["red"]))

    def test_simple_imputer_transform_before_fit_raises(self):
        imputer = SimpleImputer()

        with pytest.raises(ValueError):
            imputer.transform(np.array([[np.nan]]))

    def test_one_hot_invalid_handle_unknown_raises(self):
        with pytest.raises(ValueError):
            OneHotEncoder(handle_unknown="skip")

    def test_minmax_invalid_feature_range_type_raises(self):
        with pytest.raises(ValueError):
            MinMaxScaler(feature_range=[0, 1])

    def test_simple_imputer_unknown_strategy_raises(self):
        imputer = SimpleImputer(strategy="mode")

        with pytest.raises(ValueError):
            imputer.fit(np.array([[1.0], [2.0]]))

    def test_simple_imputer_partial_fit_feature_mismatch_raises(self):
        imputer = SimpleImputer(strategy="mean")
        imputer.partial_fit(np.array([[1.0, 2.0]]))

        with pytest.raises(ValueError):
            imputer.partial_fit(np.array([[1.0, 2.0, 3.0]]))

    def test_one_hot_partial_fit_feature_mismatch_raises(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([["red", "small"]]))

        with pytest.raises(ValueError):
            enc.partial_fit(np.array([["blue"]]))

    def test_minmax_partial_fit_feature_mismatch_raises(self):
        scaler = MinMaxScaler()
        scaler.partial_fit(np.array([[1.0, 2.0]]))

        with pytest.raises(ValueError):
            scaler.partial_fit(np.array([[1.0, 2.0, 3.0]]))