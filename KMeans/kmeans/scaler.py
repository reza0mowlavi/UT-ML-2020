import numpy as np


class StandardScaler:
    def __init__(self, epsilon=1e-14):
        """Standardize features by removing the mean and scaling to unit variance

        Args:
            epsilon (float, optional): Value to . Defaults to 1e-14.
        """
        self.epsilon = epsilon

    def fit(self, X, y=None):
        self.means = np.mean(X, axis=0, keepdims=True)
        self.var = np.var(X, axis=0, keepdims=True)
        return self

    def transform(self, X):
        return (X - self.means) / np.sqrt(self.var + self.epsilon)
