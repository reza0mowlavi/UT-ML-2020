import numpy as np


class PCA:
    def __init__(self, normalize=True, epsilon=1e-7):
        self.epsilon = epsilon
        self.normalize = True
        self._fitted = False

    def fit(self, X, y=None):
        if self.normalize:
            X = self._normalization(X)
        _, S, Vh = np.linalg.svd(X)
        V = Vh.T
        indices = np.argsort(S)[::-1]
        S = S[indices]
        self.V = V[:, indices]
        self.principal_vars = S / np.sum(S)
        self._fitted = True

    @property
    def n_non_zero_principal_component(self):
        return len(self.principal_vars)

    def transform(self, X, n_components=None):
        if self.normalize:
            X = self._normalization(X)

        if n_components is None:
            return X @ self.V

        if 0 < n_components < 1:
            cumsum = np.cumsum(self.principal_vars)
            n_components = np.argmax(cumsum >= n_components) + 1

        return X @ self.V[:, :n_components]

    def reconstruct(self, Z):
        m = np.shape(Z)[1]
        X = Z @ self.V[:, :m].T
        X = X * self.std + self.mu
        return X

    def _normalization(self, X):
        if self._fitted == False:
            self.mu = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.std = np.sqrt(var + self.epsilon)
        return (X - self.mu) / self.std

