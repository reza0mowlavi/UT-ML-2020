import numpy as np
from scipy import stats


class KNNClassifier:
    def __init__(
        self,
        n_neighbors=5,
        minkowski_order=2,
        weights="uniform",
        copy_data=False,
        epsilon=1e-7,
    ):
        """Classifier implementing the k-nearest neighbors vote.

        Args:
            n_neighbors (int, optional): Number of neighbors to use for KNN. Defaults to 5.
            minkowski_order (int, optional): Minkowski distance order. Defaults to 2.
            weights (str, optional): weight function used in prediction. Defaults to "uniform".
                    - 'uniform' : uniform weights.  All points in each neighborhood
                    are weighted equally.
                    - 'distance' : weight points by the inverse of their distance.
                    in this case, closer neighbors of a query point will have a
                    greater influence than neighbors which are further away.
            copy_data (bool, optional): Copy X,y or just keep a reference to them. Defaults to False.
            epsilon ([type], optional): Use for numerical stabality when "weight" is set to "distance".
                     Defaults to 1e-7.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.minkowski_order = minkowski_order
        self.copy_data = copy_data
        self.epsilon = epsilon

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Args:
            X (array-like): of shape (n_samples, n_features)
                            Training vectors, where n_samples is the number of samples and
                            n_features is the number of features.
            y (array-like): of shape (n_samples,)
                            Target values.

        Returns:
            Returns itself (object)
        """
        if self.copy_data:
            self.X = np.copy(X)
            self.y = np.copy(y)
        else:
            self.X = X
            self.y = y

        self.unique_labels = np.unique(y)
        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Args:
            X (array-like): of shape (n_samples, n_features)

        Returns:
            [array-like]: ndarray of shape (n_samples,)
                          Predicted target values for X
        """
        if self.weights == "uniform":
            return self._predict_uniform(X)
        if self.weights == "distance":
            return self._predict_distance(X)

    def _predict_uniform(self, X):
        _, labels = self._compute_distance(X)
        return stats.mode(labels, axis=1).mode.flatten()

    def _predict_distance(self, X):
        distances, labels = self._compute_distance(X)
        weights = 1 / (distances + self.epsilon)
        weighted_distance = np.array(
            [
                [
                    np.sum(weights[i][labels[i] == label])
                    for label in range(len(self.unique_labels))
                ]
                for i in range(np.shape(X)[0])
            ]
        )
        return np.argmax(weighted_distance, axis=1)

    def _compute_distance(self, samples):
        N = np.shape(samples)[0]
        distances = np.array(
            [
                np.linalg.norm(
                    self.X - samples[i].reshape(1, -1),
                    ord=self.minkowski_order,
                    axis=-1,
                )
                for i in range(N)
            ]
        )
        indices = np.argsort(distances, axis=-1)[:, : self.n_neighbors]
        distances = np.array([distances[i][indices[i]] for i in range(N)])
        labels = np.array([self.y[indices[i]] for i in range(N)])
        return distances, labels

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
