"""
Pipeline utilities for NumCompute-Stream.

This module builds on the lightweight Pipeline and FeatureUnion tools from the
original NumCompute package. The batch fit, transform, fit_transform, and
predict behaviour is retained because it is still useful for normal workflows
and for transforming each incoming data chunk.

Streaming support is added through partial_fit(), allowing transformers and
models to be updated one chunk at a time.
"""

import numpy as np


def _validate_named_steps(steps, name="steps"):
    """
    Validate a list of named pipeline steps.
    """
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError(f"{name} must be a non-empty list of (name, object) tuples")

    seen_names = set()

    for item in steps:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{name} must contain (name, object) tuples")

        step_name, step = item

        if not isinstance(step_name, str) or step_name == "":
            raise ValueError("step names must be non-empty strings")

        if step_name in seen_names:
            raise ValueError(f"duplicate step name found: {step_name}")

        if step is None:
            raise ValueError(f"step '{step_name}' cannot be None")

        seen_names.add(step_name)


class Pipeline:
    """
    Chain a sequence of transformer and estimator steps.

    Parameters
    ----------
    steps : list of (str, object)
        Named pipeline steps. Intermediate steps should implement transform().
        The final step may be a transformer or an estimator.
    """

    def __init__(self, steps):
        _validate_named_steps(steps)
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}

    def _fit_step(self, step, X, y=None):
        """
        Call fit with y when supported, otherwise call fit with X only.
        """
        try:
            return step.fit(X, y)
        except TypeError:
            return step.fit(X)

    def _partial_fit_step(self, step, X, y=None):
        """
        Call partial_fit when available, otherwise fall back to fit.
        """
        if hasattr(step, "partial_fit"):
            try:
                return step.partial_fit(X, y)
            except TypeError:
                return step.partial_fit(X)

        if hasattr(step, "fit"):
            return self._fit_step(step, X, y)

        return step

    def fit(self, X, y=None):
        """
        Fit all steps sequentially, passing transformed data between steps.
        """
        X_current = X

        for index, (name, step) in enumerate(self.steps):
            if hasattr(step, "fit"):
                self._fit_step(step, X_current, y)

            if index < len(self.steps) - 1:
                if not hasattr(step, "transform"):
                    raise AttributeError(
                        f"Intermediate step '{name}' must implement transform()."
                    )

                X_current = step.transform(X_current)

        return self

    def partial_fit(self, X, y=None):
        """
        Incrementally update the pipeline on one incoming data chunk.

        Intermediate steps are updated first, then used to transform the chunk.
        The final step is then updated using the transformed chunk.
        """
        X_current = X

        for index, (name, step) in enumerate(self.steps):
            is_final_step = index == len(self.steps) - 1

            if is_final_step:
                if not hasattr(step, "partial_fit") and not hasattr(step, "fit"):
                    raise AttributeError(
                        f"Final step '{name}' must implement partial_fit() or fit()."
                    )

                self._partial_fit_step(step, X_current, y)

            else:
                if not hasattr(step, "transform"):
                    raise AttributeError(
                        f"Intermediate step '{name}' must implement transform()."
                    )

                self._partial_fit_step(step, X_current, y)
                X_current = step.transform(X_current)

        return self

    def transform(self, X):
        """
        Apply transform() of every step in sequence.
        """
        X_current = X

        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise AttributeError(
                    f"Step '{name}' does not implement transform()."
                )

            X_current = step.transform(X_current)

        return X_current

    def fit_transform(self, X, y=None):
        """
        Fit and transform each step sequentially.
        """
        X_current = X

        for name, step in self.steps:
            if hasattr(step, "fit"):
                self._fit_step(step, X_current, y)

            if hasattr(step, "transform"):
                X_current = step.transform(X_current)

        return X_current

    def predict(self, X):
        """
        Transform through intermediate steps, then predict with the final step.
        """
        X_current = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise AttributeError(
                    f"Intermediate step '{name}' must implement transform()."
                )

            X_current = step.transform(X_current)

        name, final_step = self.steps[-1]

        if not hasattr(final_step, "predict"):
            raise AttributeError(
                f"Final step '{name}' does not implement predict()."
            )

        return final_step.predict(X_current)

    def get_step(self, name):
        """
        Retrieve a pipeline step by name.
        """
        if name not in self.named_steps:
            raise KeyError(f"No step named '{name}' found in Pipeline.")

        return self.named_steps[name]

    def __repr__(self):
        parts = ", ".join(
            f"('{name}', {step.__class__.__name__})"
            for name, step in self.steps
        )
        return f"Pipeline(steps=[{parts}])"


class FeatureUnion:
    """
    Concatenate outputs of multiple transformers side by side.

    Parameters
    ----------
    transformer_list : list of (str, object)
        Named transformers. Each transformer should implement fit() and
        transform().
    """

    def __init__(self, transformer_list):
        _validate_named_steps(transformer_list, name="transformer_list")
        self.transformer_list = transformer_list
        self.named_transformers = {
            name: transformer
            for name, transformer in transformer_list
        }

    def fit(self, X, y=None):
        """
        Fit all transformers independently on the same input.
        """
        for name, transformer in self.transformer_list:
            if not hasattr(transformer, "fit"):
                raise AttributeError(
                    f"Transformer '{name}' must implement fit()."
                )

            transformer.fit(X, y)

        return self

    def partial_fit(self, X, y=None):
        """
        Incrementally update each transformer when partial_fit is available.

        If a transformer does not implement partial_fit(), fit() is used as a
        fallback.
        """
        for name, transformer in self.transformer_list:
            if hasattr(transformer, "partial_fit"):
                try:
                    transformer.partial_fit(X, y)
                except TypeError:
                    transformer.partial_fit(X)
            elif hasattr(transformer, "fit"):
                transformer.fit(X, y)
            else:
                raise AttributeError(
                    f"Transformer '{name}' must implement partial_fit() or fit()."
                )

        return self

    def transform(self, X):
        """
        Transform input with each transformer and horizontally stack outputs.
        """
        parts = []

        for name, transformer in self.transformer_list:
            if not hasattr(transformer, "transform"):
                raise AttributeError(
                    f"Transformer '{name}' must implement transform()."
                )

            part = transformer.transform(X)
            parts.append(np.asarray(part))

        return np.hstack(parts)

    def fit_transform(self, X, y=None):
        """
        Fit and transform each transformer, then horizontally stack outputs.
        """
        parts = []

        for name, transformer in self.transformer_list:
            if hasattr(transformer, "fit_transform"):
                part = transformer.fit_transform(X, y)
            else:
                if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                    raise AttributeError(
                        f"Transformer '{name}' must implement fit() and transform()."
                    )

                transformer.fit(X, y)
                part = transformer.transform(X)

            parts.append(np.asarray(part))

        return np.hstack(parts)

    def get_transformer(self, name):
        """
        Retrieve a transformer by name.
        """
        if name not in self.named_transformers:
            raise KeyError(f"No transformer named '{name}' found in FeatureUnion.")

        return self.named_transformers[name]

    def __repr__(self):
        parts = ", ".join(
            f"('{name}', {transformer.__class__.__name__})"
            for name, transformer in self.transformer_list
        )
        return f"FeatureUnion(transformer_list=[{parts}])"