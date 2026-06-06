"""
Module: pipeline.py
Description: Lightweight Pipeline abstraction for chaining Transformer and Estimator
             steps. Supports fit, transform, fit_transform, and predict workflows,
             following a minimal Transformer/Estimator protocol compatible with
             the NumCompute preprocessing and modelling API.
"""
import numpy as np

class Pipeline:
    """
    Chain a sequence of Transformer and/or Estimator steps.
    steps : list of (str, object) tuples
        Named pipeline steps e.g. [('scaler', StandardScaler()), ...].
        All steps except the last must implement transform().
    """

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        """
        Fit all steps sequentially, passing transformed output between steps.

        Parameters X : np.ndarray, y : np.ndarray
        Returns self : Pipeline
        Time complexity  : O(sum of individual step fit complexities)
        Space complexity : O(n_samples * n_features)
        """
        X_current = X
        for i, (name, step) in enumerate(self.steps):
            if hasattr(step, "fit"):
                self._fit_step(step, X_current, y)
            if i < len(self.steps) - 1:
                if not hasattr(step, "transform"):
                    raise AttributeError(
                        f"Intermediate step '{name}' must implement transform()."
                    )
                X_current = step.transform(X_current)
        return self
    
    def _fit_step(self, step, X, y):
        """
        Safely call fit depending on whether the step accepts y.
        """
        try:
            return step.fit(X, y)
        except TypeError:
            return step.fit(X)

    def transform(self, X):
        """
        Apply transform() of every step in sequence.

        Parameters X : np.ndarray
        Returns X_out : np.ndarray
        Raises
        AttributeError
            If any step does not implement transform().

        Time complexity  : O(sum of individual step transform complexities)
        Space complexity : O(n_samples * n_features)
        """
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise AttributeError(
                    f"Step '{name}' does not implement transform()."
                )
            X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform each step sequentially.
        Parameters X : np.ndarray, y : np.ndarray
        Returns X_out : np.ndarray
        Time complexity  : O(sum of individual step fit_transform complexities)
        Space complexity : O(n_samples * n_features)
        """
        for name, step in self.steps:
            if hasattr(step, "fit"):
                self._fit_step(step, X, y)
            if hasattr(step, "transform"):
                X = step.transform(X)
        return X

    def predict(self, X):
        """
        Transform through all intermediate steps then call predict() on the last.
        Parameters
        X : np.ndarray
        Returns y_pred : np.ndarray

        Raises
        AttributeError
            If the final step does not implement predict().

        Time complexity  : O(sum of transform + final predict complexities)
        Space complexity : O(n_samples * n_features)
        """
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise AttributeError(
                    f"Intermediate step '{name}' must implement transform()."
                )
            X = step.transform(X)
        name, final_step = self.steps[-1]
        if not hasattr(final_step, "predict"):
            raise AttributeError(
                f"Final step '{name}' does not implement predict()."
            )
        return final_step.predict(X)
class FeatureUnion:
    """
    Concatenate outputs of multiple transformers side-by-side (column-wise).

    Each transformer is fit and applied independently to the input X,
    and their outputs are horizontally stacked into a single array.

    Parameters
    transformer_list : list of (str, transformer) tuples
        Each transformer must implement fit() and transform().

    Time complexity  : O(sum of individual transformer complexities)
    Space complexity : O(n_samples * total_output_features)
    """

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        """
        Fit all transformers independently on X.

        Parameters
        X : np.ndarray
        y : np.ndarray, optional

        Returns
        self : FeatureUnion

        Time complexity  : O(sum of individual fit complexities)
        Space complexity : O(n_samples * n_features)
        """
        for name, transformer in self.transformer_list:
            if not hasattr(transformer, "fit"):
                raise AttributeError(
                    f"Transformer '{name}' must implement fit()."
                )
            transformer.fit(X, y)
        return self

    def transform(self, X):
        """
        Apply each transformer and horizontally stack the results.

        Parameters
        X : np.ndarray

        Returns
        X_out : np.ndarray, shape (n_samples, sum of output features)

        Raises
        AttributeError
            If any transformer does not implement transform().

        Time complexity  : O(sum of individual transform complexities)
        Space complexity : O(n_samples * total_output_features)
        """
        parts = []
        for name, transformer in self.transformer_list:
            if not hasattr(transformer, "transform"):
                raise AttributeError(
                    f"Transformer '{name}' must implement transform()."
                )
            parts.append(transformer.transform(X))
        return np.hstack(parts)

    def fit_transform(self, X, y=None):
        """
        Fit and transform each transformer, then horizontally stack results.

        Parameters
        X : np.ndarray
        y : np.ndarray, optional

        Returns
        X_out : np.ndarray, shape (n_samples, sum of output features)

        Time complexity  : O(sum of individual fit_transform complexities)
        Space complexity : O(n_samples * total_output_features)
        """
        parts = []
        for name, transformer in self.transformer_list:
            if hasattr(transformer, "fit_transform"):
                parts.append(transformer.fit_transform(X, y))
            else:
                if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                    raise AttributeError(
                        f"Transformer '{name}' must implement fit() and transform()."
                    )
                transformer.fit(X, y)
                parts.append(transformer.transform(X))
        return np.hstack(parts)

    def get_transformer(self, name):
        """
        Retrieve a transformer by name.

        Parameters
        name : str

        Returns
        -------
        transformer : object

        Raises
        KeyError
            If no transformer with the given name exists.
        """
        for t_name, transformer in self.transformer_list:
            if t_name == name:
                return transformer
        raise KeyError(f"No transformer named '{name}' found in FeatureUnion.")

    def __repr__(self):
        t_str = ", ".join(f"('{n}', {t.__class__.__name__})" for n, t in self.transformer_list)
        return f"FeatureUnion(transformer_list=[{t_str}])"