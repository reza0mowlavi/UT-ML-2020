import numpy as np


class PolyNomialFeatures:
    def __init__(self, degree, include_bias=False):
        """Extract Polynomial features

        Args:
            degree : degree of polynomial features
            include_bias (bool, optional): Wheter to include bias or not. Defaults to False.

        Raises:
            ValueError: 'degree' must be greater than 1.
        """
        if degree < 1:
            raise ValueError('Parameter "degree" must be greater than 1.')
        self.degree = degree
        self.include_bias = include_bias

    def transform(self, X, y=None):
        """Extract polynomila features

        Args:
            X : Data used to extract features [None,dim]
            y (optional): It does nothing. Defaults to None.

        Returns:
            Extracted features: [None,(selection dim from n)]
        """
        results = []
        P = np.ones((len(X), 1), dtype=X.dtype)
        if self.include_bias:
            results.append(P)
        self._add(P, X, start=0, level=1, results=results)
        return np.concatenate(results, 1)

    def _add(self, P, X, start, level, results):
        if level > self.degree:
            return
        for i in range(start, np.shape(X)[1]):
            new_P = P * X[:, i].reshape(-1, 1)
            results.append(new_P)
            self._add(new_P, X, i, level + 1, results)
