import numpy as np
from .losses import SparseCrossEntropy
from .activations import Softmax


class LogisticRegression:
    def initialize(self, X, y, class_weights):
        self.loss = SparseCrossEntropy(class_weights=class_weights)
        self.activation = Softmax()
        input_dim = np.shape(X)[1]
        output_dim = len(np.unique(y))
        self.weight = np.random.normal(
            loc=0, scale=(input_dim + output_dim) / 2, size=(output_dim, input_dim)
        ).astype(dtype=X.dtype)
        self.bias = np.zeros((output_dim,), dtype=X.dtype)

    def fit(
        self,
        X,
        y,
        learning_rate=0.01,
        batch_size=32,
        epochs=100,
        early_stopping=None,
        class_weights=None,
        clip_norm=None,
        verbose=True,
        return_history=True,
    ):
        """Fit Naive Logistic Regression according to X, y

        Args:
            X (array-like): of shape (n_samples, n_features)
                            Training vectors, where n_samples is the number of samples and
                            n_features is the number of features.
            y (array-like): of shape (n_samples,)
                            Target values.
            learning_rate (float, optional): Gradient Descent learning rate.
                                             Defaults to 0.01.
            batch_size (int, optional): Defaults to 32.
            epochs (int, optional): MiniBatch training epochs. Defaults to 100.
            early_stopping (int, optional): If not specified,won't use early stopping
                                            but if given int,it would be its patience.
                                            Defaults to None.
            class_weights (dict, optional): used for weighting the loss function
                                            (during training only).
                                            This can be useful to tell the model to
                                            "pay more attention" to samples from
                                            an under-represented class. Defaults to None.
            clip_norm (float, optional): Defaults to None.
            verbose (bool, optional): If print training process or not. Defaults to True.
            return_history (bool, optional): If return loss,score over training or not.
                                             Defaults to True.

        Returns:
            if "return_history=True" then it would return
                losses,scores over training in a dict
        """
        y = y.astype("int32")
        self.initialize(X, y, class_weights)

        losses = accs = None
        if return_history:
            losses = np.empty(epochs + 1, dtype="float32")
            accs = np.empty(epochs + 1, dtype="float32")

        if early_stopping is not None:
            counter = 0
            loss = np.inf

        self.status(X, y, 0, verbose, return_history, accs, losses, early_stopping)

        for epoch in range(1, epochs + 1):
            self.fit_batch(X, y, learning_rate, batch_size, clip_norm)
            vals = self.status(
                X, y, epoch, verbose, return_history, accs, losses, early_stopping,
            )
            if early_stopping is not None:
                if vals[0] >= loss:
                    counter += 1
                loss = vals[0]
                if counter > early_stopping:
                    break

        print()
        if return_history:
            return {"loss": losses, "acc": accs}

        return self

    def status(
        self, X, y, epoch, verbose, return_history, accs, losses, early_stopping,
    ):
        if return_history or verbose or early_stopping is not None:
            y_pred = self.predict_proba(X)
            loss = np.mean(self.loss(y, y_pred))
            y_pred = np.argmax(y_pred, axis=1).flatten()
            acc = self.accuracy(y, y_pred)
            if verbose:
                print(f"\rEpoch {epoch} => loss={loss} - Acc={acc}", end="")
            if return_history:
                losses[epoch] = loss
                accs[epoch] = acc
            return loss, acc

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).flatten().astype("int64")

    def predict_proba(self, X):
        Z = X @ self.weight.T + self.bias
        return self.activation(Z)

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
        return self.accuracy(y, self.predict(X))

    def accuracy(self, y_true, y_pred):
        T = np.sum(y_true == y_pred)
        return T / np.shape(y_true)[0]

    def batch_error(self, x_batch, y_batch):
        """Compute error for given batch

        Returns:
            bias error,weight_error
        """
        D_, D = np.shape(self.weight)
        N = np.shape(x_batch)[0]

        Z = x_batch @ self.weight.T + self.bias
        y_pred = self.activation(Z, trainig=True)

        loss_error = self.loss.derivative(y_batch, y_pred)
        activation_error = self.activation.derivative()

        bias_error = np.sum(
            [
                [
                    [loss_error[n, k] * activation_error[n, k, i] for k in range(D_)]
                    for i in range(D_)
                ]
                for n in range(N)
            ],
            axis=-1,
        ).astype(self.bias.dtype)

        weight_error = np.sum(
            [
                [
                    [
                        [
                            loss_error[n, k] * activation_error[n, k, i] * x_batch[n, j]
                            for k in range(D_)
                        ]
                        for j in range(D)
                    ]
                    for i in range(D_)
                ]
                for n in range(N)
            ],
            axis=-1,
        ).astype(self.weight.dtype)

        bias_error = np.mean(bias_error, axis=0)
        weight_error = np.mean(weight_error, axis=0)

        return bias_error, weight_error

    def fit_batch(self, X, y, learning_rate, batch_size, clip_norm):
        """Using given batch improve our bias,weights
        """
        x_batch, y_batch = self._get_batch(X, y, batch_size)
        bias_error, weight_error = self.batch_error(x_batch, y_batch)
        if clip_norm:
            weight_norm = np.linalg.norm(weight_error)
            if weight_norm > clip_norm:
                weight_error = weight_error / weight_norm * clip_norm
                bias_error = bias_error / np.linalg.norm(bias_error) * clip_norm

        self.bias -= learning_rate * bias_error
        self.weight -= learning_rate * weight_error

    def _get_batch(self, X, y, batch_size):
        """Create mini batch of given size from (X,y)

        Returns:
            x_batch,y_batch
        """
        indices = np.random.randint(0, np.shape(X)[0], batch_size)
        return X[indices], y[indices]
