"""
Decision tree model for NumCompute-Stream.

This module implements a small decision tree classifier from scratch using
NumPy. The first version supports normal batch fitting and prediction.

Streaming support is added separately through partial_fit().
"""

import numpy as np


class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=3
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    criterion : {"gini", "entropy"}, default="gini"
        Split quality measure.
    max_features : int or None, default=None
        Number of features considered at each split. If None, all features are used.
    random_state : int or None, default=None
        Random seed used when feature subsampling is enabled.
    """

    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.tree_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.rng_ = np.random.default_rng(random_state)

    def fit(self, X, y):
        """
        Fit the decision tree on a full batch of data.
        """
        X, y = self._validate_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, depth=0)

        return self

    def predict(self, X):
        """
        Predict class labels for input samples.
        """
        if self.tree_ is None:
            raise ValueError("DecisionTreeClassifier is not fitted yet")

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        return np.array([self._predict_one(row, self.tree_) for row in X])

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
            raise ValueError("DecisionTreeClassifier does not accept NaN values in X")

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

    def _build_tree(self, X, y, depth):
        """
        Recursively build a decision tree.
        """
        prediction = self._majority_class(y)

        node = {
            "prediction": prediction,
            "feature_index": None,
            "threshold": None,
            "left": None,
            "right": None,
            "is_leaf": True,
        }

        if self._should_stop(X, y, depth):
            return node

        feature_index, threshold = self._best_split(X, y)

        if feature_index is None:
            return node

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return node

        node["feature_index"] = feature_index
        node["threshold"] = threshold
        node["is_leaf"] = False
        node["left"] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _should_stop(self, X, y, depth):
        """
        Decide whether a node should become a leaf.
        """
        if depth >= self.max_depth:
            return True

        if X.shape[0] < self.min_samples_split:
            return True

        if np.unique(y).size == 1:
            return True

        return False

    def _best_split(self, X, y):
        """
        Find the best feature and threshold for a split.
        """
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        parent_impurity = self._impurity(y)
        feature_indices = self._feature_indices(X.shape[1])

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gain = self._information_gain(
                    y,
                    y[left_mask],
                    y[right_mask],
                    parent_impurity,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _feature_indices(self, n_features):
        """
        Return feature indices considered for splitting.
        """
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)

        return self.rng_.choice(n_features, size=self.max_features, replace=False)

    def _information_gain(self, parent, left, right, parent_impurity):
        """
        Compute impurity reduction from a split.
        """
        n_total = parent.size
        left_weight = left.size / n_total
        right_weight = right.size / n_total

        child_impurity = (
            left_weight * self._impurity(left)
            + right_weight * self._impurity(right)
        )

        return parent_impurity - child_impurity

    def _impurity(self, y):
        """
        Compute node impurity using the selected criterion.
        """
        if self.criterion == "gini":
            return self._gini(y)

        if self.criterion == "entropy":
            return self._entropy(y)

        raise ValueError("criterion must be 'gini' or 'entropy'")

    def _gini(self, y):
        """
        Compute Gini impurity.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        return 1.0 - np.sum(probabilities ** 2)

    def _entropy(self, y):
        """
        Compute entropy impurity.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]

        return -float(np.sum(probabilities * np.log2(probabilities)))

    def _majority_class(self, y):
        """
        Return the most common class label.

        Ties are resolved by NumPy's sorted unique order.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict_one(self, row, node):
        """
        Predict one sample by walking through the tree.
        """
        while not node["is_leaf"]:
            feature_index = node["feature_index"]
            threshold = node["threshold"]

            if row[feature_index] <= threshold:
                node = node["left"]
            else:
                node = node["right"]

        return node["prediction"]