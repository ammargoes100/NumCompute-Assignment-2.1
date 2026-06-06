# numcompute/preprocessing.py
"""
preprocessing.py - Data preprocessing utilities for NumCompute.
Provides:
- StandardScaler: z-score standardization (zero mean, unit variance)
- MinMaxScaler: scale features to a given range (default [0,1])
- OneHotEncoder: one-hot encoding for categorical variables
- SimpleImputer: replace NaN with a constant
"""

import numpy as np

class StandardScaler:
    """(X - mean) / std, per feature."""

    def __init__(self, with_mean=True, with_std=True):
        """Initialize the scaler"""
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def _ensure_2d(self, X):  # helper: convert 1D array to column vector
        return X.reshape(-1, 1) if X.ndim == 1 else X
    

    def _validate_non_empty(self, X, method="fit"):
        """Raise ValueError if X has zero samples."""
        if X.shape[0] == 0:
            raise ValueError(f"X has 0 samples; cannot {method}.")

    def fit(self, X):
        """
        Parameters:
            X : ndarray, shape (n_samples, n_features)

        Returns:
            self

        Raises:
            ValueError : if X has zero rows.

        Time complexity : O(n_samples * n_features)
        Space complexity : O(n_features)
        """
        # Convert 1D input to 2D column vector
        X = self._ensure_2d(X)
        if X.ndim != 2:   # after reshape, should be 2D; if still not, it's >2
            raise ValueError(f"Input must be 2D array, got {X.ndim}D")
        self._validate_non_empty(X, "fit")

        # Store number of features after possible reshape
        self.n_features_in_ = X.shape[1]

        # Compute mean 
        if self.with_mean:
            # np.mean with axis=0 gives column-wise means
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = None

        # Compute standard deviation 
        if self.with_std:
            # np.std with axis=0, ddof=0 -> population std
            self.scale_ = np.std(X, axis=0, ddof=0)
            # If a feature has zero variance, set scale to 1 to avoid division by zero
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_ = None

        return self

    def transform(self, X):
        """
        Parameters:
            X : ndarray, shape (n_samples, n_features)

        Returns:
            X_scaled : ndarray, same shape

        Raises:
            ValueError : if number of features does not match fit.

        Time complexity : O(n_samples * n_features)
        Space complexity : O(n_samples * n_features)  # copy
        """
        # Ensure input is 2D (if 1D, make column)
        X = self._ensure_2d(X)

        # Check that number of features matches fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        X_scaled = X.astype(float)

        # Center if needed
        if self.with_mean:
            X_scaled -= self.mean_

        # Scale if needed
        if self.with_std:
            X_scaled /= self.scale_

        return X_scaled

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class MinMaxScaler:
    """Scale features to [min, max] range, default (0,1)."""
    def __init__(self, feature_range=(0, 1)):
        if feature_range[0] > feature_range[1]:
            raise ValueError(f"min must be <= max, got {feature_range}")
        self.feature_range = feature_range
        # self.min_ = None
        # self.max_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_min_for_adjust = None 
        self.n_features_in_ = None

    def _ensure_2d(self, X):
        """If X is 1D, reshape to (n_samples, 1)."""
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X
    
    def _validate_non_empty(self, X, method="fit"):
        """Raise ValueError if X has zero samples."""
        if X.shape[0] == 0:
            raise ValueError(f"X has 0 samples; cannot {method}.")

    def fit(self, X):
        """
        Parameters:
            X : ndarray, shape (n_samples, n_features)

        Returns:
            self

        Raises:
            ValueError : if X has zero rows.

        Time complexity : O(n_samples * n_features)
        Space complexity : O(n_features)
        """
        X = self._ensure_2d(X)
        self._validate_non_empty(X, "fit")
        self.n_features_in_ = X.shape[1]

        # Store original data min and max per feature
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)

        # Compute scaling factor
        range_min, range_max = self.feature_range
        scale_numerator = range_max - range_min
        data_range = self.data_max_ - self.data_min_

        self.scale_ = np.zeros_like(data_range, dtype=float)
        non_const = data_range != 0
        self.scale_[non_const] = scale_numerator / data_range[non_const]

        self.data_min_for_adjust = range_min

        return self

    def transform(self, X):
        """
        Parameters:
            X : ndarray, shape (n_samples, n_features)

        Returns:
            X_scaled : ndarray, same shape

        Raises:
            ValueError : if number of features does not match fit.

        Time complexity : O(n_samples * n_features)
        Space complexity : O(n_samples * n_features)
        """
        X = self._ensure_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        X_scaled = (X - self.data_min_) * self.scale_ + self.data_min_for_adjust

        return X_scaled

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

class OneHotEncoder:
    """One-hot encoding with handle_unknown = 'error' or 'ignore'."""

    def __init__(self, sparse_output=False, handle_unknown='error'):
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.n_features_in_ = None
        self.n_categories_ = None

    def _ensure_2d(self, X):
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def _check_empty(self, X, stage="fit"):
        if X.shape[0] == 0:
            raise ValueError(f"X has 0 samples; cannot {stage}")

    def _check_nan(self, X):
        """Raise ValueError if array is numeric and contains NaN."""
        if np.issubdtype(X.dtype, np.number) and np.any(np.isnan(X)):
            raise ValueError("OneHotEncoder does not accept NaN values")

    def fit(self, X):
        """
            Parameters:
                X : np.ndarray, shape (n_samples, n_features)
                    Categorical values (integers or strings).

            Returns:
                self

            Raises:
                ValueError : if X has zero rows.

            Time complexity : O(n_samples * n_features) + O(unique per feature)
            Space complexity : O(sum of unique categories)
        """
        X = self._ensure_2d(X)
        self._check_empty(X, "fit")
        self._check_nan(X)
        self.n_features_in_ = X.shape[1]
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        self.n_categories_ = [len(c) for c in self.categories_]
        return self

    def transform(self, X):
        """
        Parameters:
            X : np.ndarray, shape (n_samples, n_features)

        Returns:
            X_out : np.ndarray, shape (n_samples, total_categories), dtype=int8

        Raises:
            ValueError : if number of features differs, or unknown category when handle_unknown='error'.

        Time complexity : O(n_samples * total_categories)
        Space complexity : O(n_samples * total_categories)
        """
        X = self._ensure_2d(X)
        self._check_nan(X)   # check before processing
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        n_samples = X.shape[0]
        total_cols = sum(self.n_categories_)
        out = np.zeros((n_samples, total_cols), dtype=np.int8)
        col_start = 0

        for i, cats in enumerate(self.categories_):
            n_cats = len(cats)
            match = (X[:, i][:, None] == cats).astype(np.int8)

            if self.handle_unknown == 'error':
                unknown = (match.sum(axis=1) == 0)
                if unknown.any():
                    bad = X[:, i][np.where(unknown)[0][0]]
                    raise ValueError(f"Unknown category '{bad}' in feature {i}")

            out[:, col_start:col_start + n_cats] = match
            col_start += n_cats

        if self.sparse_output:
            import warnings
            warnings.warn("sparse_output ignored – returning dense")
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace missing values with a constant, mean, or median per feature."""

    def __init__(self, strategy='constant', fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self.n_features_in_ = None

    def _ensure_2d(self, X):
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def _check_empty(self, X, stage="fit"):
        if X.shape[0] == 0:
            raise ValueError(f"X has 0 samples; cannot {stage}")

    def fit(self, X):
        X = self._ensure_2d(X)
        self._check_empty(X, "fit")
        self.n_features_in_ = X.shape[1]

        if self.strategy == 'constant':
            self.statistics_ = np.full(self.n_features_in_, self.fill_value)
        elif self.strategy == 'mean':
            non_nan_count = np.count_nonzero(~np.isnan(X), axis=0)
            col_sum = np.nansum(X, axis=0)
            self.statistics_ = np.divide(col_sum, non_nan_count,
                                         out=np.zeros_like(col_sum, dtype=float),
                                         where=non_nan_count != 0)
        elif self.strategy == 'median':
            self.statistics_ = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                col = X[:, i]
                if np.all(np.isnan(col)):
                    self.statistics_[i] = 0
                else:
                    self.statistics_[i] = np.nanmedian(col)

        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")
        return self

    def transform(self, X):
        X = self._ensure_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        out = X.copy()
        nan_mask = np.isnan(out)
        out[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)