import numpy as np
from .lda import LinearDiscriminatorAnalysis
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score


class LinearDiscriminatorClassifier(LinearDiscriminatorAnalysis):
    def __init__(self, n_components):
        """LinearDiscriminatorClassifier base on Multivariate Normal dist

        Args:
            n_components (int): Number of components for
            dimensionality reduction
        """
        super().__init__(n_components)
        self._dists = None

    def fit(self, X, y):
        """
        Args:
            X : array-like of shape (n_samples, n_features)
                Training data.

            y : array-like of shape (n_samples,)
                Target values.
        """
        super().fit(X, y)
        self._fit(X, y)

    def _fit(self, X, y):
        proj = self.transform(X)

        log_priors = {}
        means = {}
        covs = {}

        data, labels = self._get_dataset(proj, y)
        for d, l in zip(data, labels):
            log_priors[l] = np.log(len(d) / len(X))
            means[l] = np.mean(d, axis=0)
            covs[l] = np.cov(d, rowvar=False)

        self._dists = {
            l: multivariate_normal(means[l], covs[l], allow_singular=True)
            for l in labels
        }
        self._log_priors = log_priors
        self._labels = np.sort(labels)

    def log_probability(self, X):
        """Compute log probablity for each classes

        Args:
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns:
            ndarray of shape (n_samples, n_components):
        """
        X = self.transform(X)

        log_proba = np.empty((len(X), len(self._labels)))
        dists = self._dists
        log_priors = self._log_priors
        log_proba = np.array(
            [
                [dists[l].logpdf(x).sum() + log_priors[l] for l in self._labels]
                for x in X
            ]
        )
        return log_proba

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns:
            ndarray of shape (n_samples,):
        """
        log_proba = self.log_probability(X)
        return np.argmax(log_proba, 1).flatten()

    def accuracy(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X : array-like of shape (n_samples, n_features)
                Test samples.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                True labels for X.

        Returns:
            float:accuracy
        """
        return accuracy_score(y, self.predict(X))
