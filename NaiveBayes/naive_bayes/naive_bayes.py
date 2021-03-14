import numpy as np


class NaiveBayes:
    def __init__(self, dists, laplace_factor=1, unknown=True, unknown_val=None):
        """Naive Bayes classifier

        Args:
            dists : list of dist. for each of features
                    dist. for all features
            laplace_factor : Defaults to 1.
            unknown : To consider if we may face values that's not in training data . Defaults to True.
        """
        self.dists = []
        if isinstance(dists, str):
            if dists == "gaussian":
                post_dist = UnivariateGaussian
            elif dists == "multinomial":
                post_dist = multinomila(laplace_factor, unknown, unknown_val)
            else:
                raise Exception(f"Specified dist. `{dists}` is unknown.")
            self.dists = post_dist
            return
        for dist in dists:
            if dist == "gaussian":
                post_dist = UnivariateGaussian
            elif dist == "multinomial":
                post_dist = Multinomial
            else:
                raise Exception(f"Specified dist. `{dist}` is unknown.")
            self.dists.append(post_dist)

    def fit(self, X, y):
        """Fit Naive Bayes classifier according to X, y

        Args:
            X (array-like): of shape (n_samples, n_features)
                            Training vectors, where n_samples is the number of samples and
                            n_features is the number of features.
            y (array-like): of shape (n_samples,)
                            Target values.

        Returns:
            self : object
        """
        data, labels = self._get_dataset(X, y)
        self.log_priors = self.compute_log_prior(data, labels, np.shape(X)[0])
        dists = (
            self.dists
            if isinstance(self.dists, list)
            else [self.dists] * np.shape(X)[1]
        )
        self.liklihood_dists = self.compute_liklihood_dist(data, labels, dists)
        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Args:
            X (array-like): of shape (n_samples, n_features)

        Returns:
            [array-like]: ndarray of shape (n_samples,)
                          Predicted target values for X
        """
        probs = self.predict_log_liklihood(X)
        return np.argmax(probs, axis=1).flatten().astype("int64")

    def predict_log_liklihood(self, X):
        """Return log-probability estimates for the test vector X.

        Args:
            X ([array-like]): of shape (n_samples, n_features)

        Returns:
            [array-like]: array-like of shape (n_samples, n_classes)
                        Returns the log-probability of the samples for each class in
                        the model. The columns correspond to the classes in sorted
                        order, as they appear in the attribute
        """
        log_liklihood = np.sum(
            [
                [
                    dist.log_liklihood(X[:, i])
                    for i, dist in enumerate(self.liklihood_dists[label])
                ]
                for label in self.log_priors.keys()
            ],
            axis=1,
        )
        log_liklihood = log_liklihood.T
        log_priors = np.array(list(self.log_priors.values())).reshape((1, -1))

        return log_liklihood + log_priors

    def _get_dataset(self, X, y):
        data = []
        labels = np.unique(y)
        for label in labels:
            data.append(X[y == label])
        return data, labels.astype("int64")

    def compute_log_prior(self, data, labels, N):
        log_prior = {
            label: np.log(np.shape(d)[0] / N) for d, label in zip(data, labels)
        }
        return log_prior

    def compute_liklihood_dist(self, data, labels, dists):
        liklihood_dists = {}
        for d, label in zip(data, labels):
            liklihood_dists[label] = []
            for i, dist in enumerate(dists):
                dist = dist()
                dist.fit(d[:, i])
                liklihood_dists[label].append(dist)
        return liklihood_dists

    def accuracy(self, y_true, y_pred):
        T = np.sum(y_true == y_pred)
        return T / np.shape(y_true)[0]

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Args:
            X (array-like): of shape (n_samples, n_features)
                            Training vectors, where n_samples is the number of samples and
                            n_features is the number of features.
            y (array-like): of shape (n_samples,)
                            Target values.

        Returns:
            float: Accuracy
        """
        pred = self.predict(X)
        return self.accuracy(y, pred)


class UnivariateGaussian:
    def fit(self, features):
        self.mean = np.mean(features)
        self.var = np.var(features)
        return self

    def log_liklihood(self, features):
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * np.log(self.var)
            - np.square(features - self.mean) / (2 * self.var)
        )


class Multinomial:
    def __init__(self, laplace_factor=0, unknown=True, unknown_val=None):
        self.laplace_factor = laplace_factor
        self.unknown_val = unknown_val
        self.unknown = unknown

    def fit(self, features):
        unique_vals = set(list(np.unique(features)))
        self.N_y = np.shape(features)[0]
        self.params = {}
        for unique_val in unique_vals:
            self.params[unique_val] = np.sum(features == unique_val)

        if self.unknown:
            get = lambda x: self.params.get(x, 0)
            self.D = len(unique_vals) + 1
        else:
            get = self.params.__getitem__
            self.D = len(unique_vals)

        self.get = np.vectorize(get)
        return self

    def log_liklihood(self, features):
        N_x_y = self.get(features)
        return (N_x_y + self.laplace_factor) / (self.N_y + self.D * self.laplace_factor)


def multinomila(laplace_factor, unknown, unknown_val):
    return lambda: Multinomial(laplace_factor, unknown, unknown_val)
