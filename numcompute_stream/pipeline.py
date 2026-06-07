import numpy as np
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
        self.steps = steps

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
        self.transformer_list = transformer_list

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

            parts.append(transformer.transform(X))

        return np.hstack(parts)

    def fit_transform(self, X, y=None):
        """
        Fit and transform each transformer, then horizontally stack outputs.
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
        """
        for transformer_name, transformer in self.transformer_list:
            if transformer_name == name:
                return transformer

        raise KeyError(f"No transformer named '{name}' found in FeatureUnion.")

    def __repr__(self):
        parts = ", ".join(
            f"('{name}', {transformer.__class__.__name__})"
            for name, transformer in self.transformer_list
        )
        return f"FeatureUnion(transformer_list=[{parts}])"