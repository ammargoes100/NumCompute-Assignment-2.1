"""
Tree ensemble models for NumCompute-Stream.

This module implements a small Random Forest / Bagging-style classifier from
scratch using NumPy and the local DecisionTreeClassifier.

The first version supports normal batch fitting and prediction. Streaming
support is added separately through partial_fit().
"""

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


class EnsembleClassifier:
    """
    Random Forest / Bagging-style ensemble classifier.

    Parameters
    ----------
    n_estimators : int, default=5
        Number of decision trees in the ensemble.
    max_depth : int, default=3
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split a tree node.
    criterion : {"gini", "entropy"}, default="gini"
        Split quality measure used by each tree.
    max_features : int or None, default=None
        Number of features considered by each tree at each split.
    bootstrap : bool, default=True
        Whether to train each tree on a bootstrap sample.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators=5,
        max_depth=3,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self.rng_ = np.random.default_rng(random_state)
        self._X_seen = None
        self._y_seen = None

    def fit(self, X, y):
        """
        Fit all trees on the given batch of data.
        """
        X, y = self._validate_X_y(X, y)

        self._X_seen = X.copy()
        self._y_seen = y.copy()

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []

        for i in range(self.n_estimators):
            X_sample, y_sample = self._sample_training_data(X, y)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=self._tree_seed(i),
            )

            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

        return self
    
    def partial_fit(self, X, y):
        """
        Update the ensemble using one incoming data chunk.

        The implementation stores chunks seen so far and refits each tree using
        bootstrap samples from the accumulated stream. This provides a simple
        streaming interface while keeping the ensemble deterministic and easy
        to inspect.
        """
        X, y = self._validate_X_y(X, y)

        if self._X_seen is None:
            self._X_seen = X.copy()
            self._y_seen = y.copy()
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features, got {X.shape[1]}"
                )

            self._X_seen = np.vstack([self._X_seen, X])
            self._y_seen = np.concatenate([self._y_seen, y])

        self.classes_ = np.unique(self._y_seen)
        self.n_features_in_ = self._X_seen.shape[1]

        self._fit_estimators_from_seen_data()

        return self

    def predict(self, X):
        """
        Predict class labels using majority voting across trees.
        """
        if not self.estimators_:
            raise ValueError("EnsembleClassifier is not fitted yet")

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        all_predictions = np.array([
            tree.predict(X)
            for tree in self.estimators_
        ])

        return self._majority_vote(all_predictions)

    def _validate_X_y(self, X, y):
        """
        Validate training data.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        if np.isnan(X).any():
            raise ValueError("EnsembleClassifier does not accept NaN values in X")

        return X, y

    def _validate_X(self, X):
        """
        Validate prediction data.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array")

        return X
    def _fit_estimators_from_seen_data(self):
        """
        Fit all trees using the accumulated streaming data.
        """
        self.estimators_ = []

        for i in range(self.n_estimators):
            X_sample, y_sample = self._sample_training_data(
                self._X_seen,
                self._y_seen,
            )

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=self._tree_seed(i),
            )

            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)
    def _sample_training_data(self, X, y):
        """
        Sample training data for one tree.
        """
        n_samples = X.shape[0]

        if not self.bootstrap:
            return X, y

        indices = self.rng_.integers(0, n_samples, size=n_samples)

        return X[indices], y[indices]

    def _majority_vote(self, all_predictions):
        """
        Majority vote across estimators.

        all_predictions has shape (n_estimators, n_samples).
        """
        voted = []

        for sample_predictions in all_predictions.T:
            values, counts = np.unique(sample_predictions, return_counts=True)
            voted.append(values[np.argmax(counts)])

        return np.array(voted)

    def _tree_seed(self, index):
        """
        Generate a deterministic seed for each tree.
        """
        if self.random_state is None:
            return None

        return self.random_state + index