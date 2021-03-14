import numpy as np


class Perceptron:
    def __init__(self,):
        self._initialized = False

    def _initialize(self, X):
        dim = np.shape(X)[1]
        self._weight = np.random.normal(loc=0.0, scale=1.0, size=(dim,)).astype(X.dtype)
        self._bias = np.zeros((1,), dtype=X.dtype)
        self._initialized = True

    def _batch_error(self, x_batch, y_batch):
        """Compute error for given batch

        Returns:
            bias error,weight_error
        """
        y_pred = self.predict(x_batch)
        error = y_pred - y_batch
        bias_error = np.mean(error, axis=0)
        weight_error = np.mean(error.reshape(-1, 1) * x_batch, axis=0)
        return bias_error, weight_error

    def _fit_batch(self, x_batch, y_batch, learning_rate):
        """Using given batch improve our bias,weights
        """
        bias_error, weight_error = self._batch_error(x_batch, y_batch)
        self._bias -= learning_rate * bias_error
        self._weight -= learning_rate * weight_error

    def decision_function(self, X):
        """Return decision value of given data

        Args:
            X : Must be 2-D array (num_samples,num_dimentions)

        Returns:
            Decision value for given input in 1-D array (num_samples,)
        """
        return X @ self._weight + self._bias

    def fit(
        self, X, y, learning_rate, epochs, batch_size=None, print_results=True,
    ):
        """
        Train perceptron using X,y and prints out accuracy in
        process. Returns sequence of accuracies over
        each epoch.

        Args:
            X : Must be 2-D array (num_samples,num_dimentions)
            y : Label of X. Must be 2-D array (num_samples,)
            learning_rate : learning rate of perceptron algorithm
            epochs : number of training iteration
            batch_size (optional): Number of sample for training
                    in each epoch. If set to 'None' algorithm use
                    all data for traninig. Defaults to None.

        Returns:
            1-D array (num_samples,):  sequence of accuracies overeach epoch.
        """
        y = y.reshape((-1,))

        if not self._initialized:
            self._initialize(X)

        accs = np.empty(epochs + 1, dtype="float32")
        accs[0] = self.score(X, y)
        print(f"Epoch {0} => Accuracy={accs[-1]}", end="")

        for epoch in range(1, epochs + 1):
            if batch_size:
                x_batch, y_batch = self._get_batch(X, y, batch_size)
            else:
                x_batch, y_batch = X, y

            self._fit_batch(x_batch, y_batch, learning_rate)

            accs[epoch] = self.score(X, y)
            if print_results:
                print(f"\rEpoch {epoch} => Accuracy={accs[epoch]}", end="")
        print()
        return accs

    def _get_batch(self, X, y, batch_size):
        """Create mini batch of given size from (X,y)

        Returns:
            x_batch,y_batch
        """
        indices = np.random.randint(0, len(X), batch_size)
        return X[indices], y[indices]

    def _get_batch1(self, X, y, batch_size):
        indices = np.arange(len(X))
        start = 0
        np.random.shuffle(indices)
        while True:
            if start >= len(X):
                start = 0
                np.random.shuffle(indices)
            yield X[indices[start : start + batch_size]], y[
                indices[start : start + batch_size]
            ]
            start += batch_size

    def predict(self, X):
        """Predict classes for given input X.

        Args:
            X : Must be 2-D array (num_samples,num_dimentions)

        Returns:
            1-D array (num_samples,): Classes for given input X
        """
        y_pred = self.decision_function(X)
        return np.where(y_pred > 0, 1, 0)

    def score(self, X, y):
        """Compute Accuracy for given X and it's labels y

        Args:
            X : Must be 2-D array (num_samples,num_dimentions)
            y : 1-D array (num_samples,) Classes for given input X

        Returns:
            Accuracy over X,y
        """
        y_pred = self.predict(X)
        T = np.sum(y == y_pred)
        return T / len(X)

