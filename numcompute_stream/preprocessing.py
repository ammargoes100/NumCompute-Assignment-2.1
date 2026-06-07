"""
Preprocessing utilities for NumCompute-Stream.

This module builds on the preprocessing transformers from the original
NumCompute package. The batch versions of StandardScaler, MinMaxScaler,
OneHotEncoder, and SimpleImputer are retained because they are still useful for
normal preprocessing and for transforming individual data chunks.

Streaming support is added through partial_fit(), allowing preprocessing state
to be updated one chunk at a time.
"""

import warnings
import numpy as np


def _ensure_2d(X):
    """
    Convert input to a 2D NumPy array.

    A 1D array is treated as a single feature column.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        return X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError(f"input must be 1D or 2D, got {X.ndim}D")

    return X


def _check_non_empty(X, stage="fit"):
    """
    Check that X has at least one sample.
    """
    if X.shape[0] == 0:
        raise ValueError(f"X has 0 samples; cannot {stage}.")


def _check_is_fitted(value, name):
    """
    Raise an error when a transformer is used before fitting.
    """
    if value is None:
        raise ValueError(f"{name} is not fitted yet")


class StandardScaler:
    """
    Standardise features using z-score scaling.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0

    def fit(self, X):
        """
        Fit the scaler on a full batch of data.
        """
        self.n_samples_seen_ = 0
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """
        Update running mean and variance using one data chunk.
        """
        X = _ensure_2d(X).astype(float)
        _check_non_empty(X, "partial_fit")

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = 0
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.var_ = np.zeros(X.shape[1], dtype=float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        chunk_count = X.shape[0]
        chunk_mean = np.mean(X, axis=0)
        chunk_var = np.var(X, axis=0)

        if self.n_samples_seen_ == 0:
            new_mean = chunk_mean
            new_var = chunk_var
            new_count = chunk_count
        else:
            old_count = self.n_samples_seen_
            new_count = old_count + chunk_count
            delta = chunk_mean - self.mean_

            new_mean = self.mean_ + delta * chunk_count / new_count
            new_var = (
                old_count * self.var_
                + chunk_count * chunk_var
                + (delta ** 2) * old_count * chunk_count / new_count
            ) / new_count

        self.n_samples_seen_ = int(new_count)
        self.mean_ = new_mean if self.with_mean else None
        self.var_ = new_var if self.with_std else None

        if self.with_std:
            self.scale_ = np.sqrt(self.var_)
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_ = None

        return self

    def transform(self, X):
        """
        Transform data using the fitted mean and scale.
        """
        _check_is_fitted(self.n_features_in_, "StandardScaler")

        X = _ensure_2d(X).astype(float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        X_scaled = X.copy()

        if self.with_mean:
            X_scaled -= self.mean_

        if self.with_std:
            X_scaled /= self.scale_

        return X_scaled

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        """
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Scale each feature to a selected range.
    """

    def __init__(self, feature_range=(0, 1)):
        if (
            not isinstance(feature_range, tuple)
            or len(feature_range) != 2
            or feature_range[0] > feature_range[1]
        ):
            raise ValueError("feature_range must be a tuple (min, max) with min <= max")

        self.feature_range = feature_range
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0

    def fit(self, X):
        """
        Fit the scaler on a full batch of data.
        """
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """
        Update running feature minimum and maximum using one data chunk.
        """
        X = _ensure_2d(X).astype(float)
        _check_non_empty(X, "partial_fit")

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.data_min_ = np.min(X, axis=0)
            self.data_max_ = np.max(X, axis=0)
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features, got {X.shape[1]}."
                )

            self.data_min_ = np.minimum(self.data_min_, np.min(X, axis=0))
            self.data_max_ = np.maximum(self.data_max_, np.max(X, axis=0))

        self.n_samples_seen_ += int(X.shape[0])

        range_min, range_max = self.feature_range
        data_range = self.data_max_ - self.data_min_

        self.scale_ = np.zeros_like(data_range, dtype=float)
        non_constant = data_range != 0
        self.scale_[non_constant] = (range_max - range_min) / data_range[non_constant]

        return self

    def transform(self, X):
        """
        Transform data using the fitted minimum and maximum.
        """
        _check_is_fitted(self.n_features_in_, "MinMaxScaler")

        X = _ensure_2d(X).astype(float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        range_min, _ = self.feature_range
        return (X - self.data_min_) * self.scale_ + range_min

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        """
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    One-hot encode categorical variables.

    Unknown categories can either raise an error or be ignored.
    """

    def __init__(self, sparse_output=False, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")

        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.n_features_in_ = None
        self.n_categories_ = None

    def _check_nan(self, X):
        """
        Numeric arrays with NaN are rejected.
        """
        if np.issubdtype(X.dtype, np.number) and np.any(np.isnan(X)):
            raise ValueError("OneHotEncoder does not accept NaN values")

    def fit(self, X):
        """
        Fit categories from a full batch of data.
        """
        self.categories_ = None
        self.n_features_in_ = None
        self.n_categories_ = None
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """
        Update known categories using one data chunk.
        """
        X = _ensure_2d(X)
        _check_non_empty(X, "partial_fit")
        self._check_nan(X)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features, got {X.shape[1]}"
                )

            for i in range(X.shape[1]):
                self.categories_[i] = np.unique(
                    np.concatenate([self.categories_[i], np.unique(X[:, i])])
                )

        self.n_categories_ = [len(categories) for categories in self.categories_]
        return self

    def transform(self, X):
        """
        Transform categorical data into one-hot encoded columns.
        """
        _check_is_fitted(self.categories_, "OneHotEncoder")

        X = _ensure_2d(X)
        self._check_nan(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        n_samples = X.shape[0]
        total_cols = sum(self.n_categories_)
        out = np.zeros((n_samples, total_cols), dtype=np.int8)

        col_start = 0

        for feature_index, categories in enumerate(self.categories_):
            n_categories = len(categories)
            match = (X[:, feature_index][:, None] == categories).astype(np.int8)

            if self.handle_unknown == "error":
                unknown = match.sum(axis=1) == 0

                if unknown.any():
                    bad = X[:, feature_index][np.where(unknown)[0][0]]
                    raise ValueError(
                        f"Unknown category '{bad}' in feature {feature_index}"
                    )

            out[:, col_start:col_start + n_categories] = match
            col_start += n_categories

        if self.sparse_output:
            warnings.warn("sparse_output ignored; returning dense array")

        return out

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        """
        return self.fit(X).transform(X)


class SimpleImputer:
    """
    Replace missing values with a constant, mean, or median per feature.
    """

    def __init__(self, strategy="constant", fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        self._sum_ = None
        self._count_ = None
        self._stored_chunks_ = []

    def fit(self, X):
        """
        Fit imputation values from a full batch of data.
        """
        self.statistics_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        self._sum_ = None
        self._count_ = None
        self._stored_chunks_ = []
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """
        Update imputation statistics using one data chunk.
        """
        X = _ensure_2d(X).astype(float)
        _check_non_empty(X, "partial_fit")

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        self.n_samples_seen_ += int(X.shape[0])

        if self.strategy == "constant":
            self.statistics_ = np.full(self.n_features_in_, self.fill_value)

        elif self.strategy == "mean":
            chunk_sum = np.nansum(X, axis=0)
            chunk_count = np.count_nonzero(~np.isnan(X), axis=0)

            if self._sum_ is None:
                self._sum_ = np.zeros(self.n_features_in_, dtype=float)
                self._count_ = np.zeros(self.n_features_in_, dtype=int)

            self._sum_ += chunk_sum
            self._count_ += chunk_count

            self.statistics_ = np.divide(
                self._sum_,
                self._count_,
                out=np.zeros_like(self._sum_, dtype=float),
                where=self._count_ != 0,
            )

        elif self.strategy == "median":
            self._stored_chunks_.append(X.copy())
            all_data = np.vstack(self._stored_chunks_)
            self.statistics_ = np.zeros(self.n_features_in_, dtype=float)

            for i in range(self.n_features_in_):
                col = all_data[:, i]

                if np.all(np.isnan(col)):
                    self.statistics_[i] = 0.0
                else:
                    self.statistics_[i] = np.nanmedian(col)

        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

        return self

    def transform(self, X):
        """
        Replace missing values using fitted statistics.
        """
        _check_is_fitted(self.statistics_, "SimpleImputer")

        X = _ensure_2d(X).astype(float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        out = X.copy()
        nan_mask = np.isnan(out)

        if nan_mask.any():
            out[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])

        return out

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        """
        return self.fit(X).transform(X)