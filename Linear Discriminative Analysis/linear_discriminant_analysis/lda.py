import numpy as np


class LinearDiscriminatorAnalysis:
    def __init__(self, n_components):
        """LinearDiscriminatorAnalysis

        Args:
            n_components (int): Number of components for
            dimensionality reduction
        """
        self.n_components = n_components
        self._w = None

    def fit(self, X, y):
        """
        Args:
            X : array-like of shape (n_samples, n_features)
                Training data.

            y : array-like of shape (n_samples,)
                Target values.

        Returns:
            float: J(w)
        """
        data, _ = self._get_dataset(X, y)
        Sb, means = self._compute_between_scatter_matrix(data)
        Sw = self._compute_within_scatter_matrix(data, means)
        Z = np.linalg.pinv(Sw) @ Sb
        eigval, eigvec = np.linalg.eig(Z)
        indices = np.argsort(eigval)[-self.n_components :]
        self._w = eigvec[:, indices]
        return self._criterion(Sb, Sw)

    def transform(self, X):
        """Project data to maximize class separation.

        Args:
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns:
            ndarray of shape (n_samples, n_components):
        """
        if self._w is None:
            raise Exception("Model hasn't fit yet.")
        return X @ self._w

    def criterion(self, X, y):
        """Compute J(w)

        Args:
            X : array-like of shape (n_samples, n_features)
                Training data.

            y : array-like of shape (n_samples,)
                Target values.

        Returns:
            float: J(w)
        """
        data, _ = self._get_dataset(X, y)
        Sb, means = self._compute_between_scatter_matrix(data)
        Sw = self._compute_within_scatter_matrix(data, means)
        return self._criterion(Sb, Sw)

    def _criterion(self, Sb, Sw):
        return np.mean((self._w.T @ Sb @ self._w) / (self._w.T @ Sw @ self._w))

    def _get_dataset(self, X, y):
        data = []
        labels = np.unique(y)
        for label in labels:
            data.append(X[y == label])
        return data, labels

    def _compute_between_scatter_matrix(self, data):
        d = np.shape(data[0])[1]
        global_mean = np.mean(np.vstack(data), 0).reshape(d, 1)

        Sb = np.zeros((d, d))
        means = []
        for X in data:
            m = np.mean(X, 0).reshape(d, 1)
            means.append(m)
            distance = m - global_mean
            Sb += len(X) * (distance @ distance.T)
        return Sb, means

    def _compute_within_scatter_matrix(self, data, means):
        d = np.shape(data[0])[1]
        Sw = np.zeros((d, d))
        for m, X in zip(means, data):
            m = m.reshape(1, -1)
            d = X - m
            Sw += d.T @ d
        return Sw
