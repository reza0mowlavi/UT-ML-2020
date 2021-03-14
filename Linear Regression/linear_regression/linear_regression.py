import numpy as np
from .losses import MSELoss
from .losses import HuberLoss
from .regularizer import L2
from .polynomial_feature import PolyNomialFeatures


class LinearRegressionBase:
    def __init__(self, loss_fn, regularizer=None, degree=None):
        self._initialized = False
        self._weight = None
        self._bias = None
        self.degree = degree
        if degree is not None:
            self.poly = PolyNomialFeatures(degree=degree, include_bias=False)
        self.loss_fn = loss_fn
        self.regularizer = regularizer

    def _fit(self, X, y):
        raise Exception("Not Implemented.")

    def fit(self, X, y):
        """Fit the X , y to the model

        Args:
            X : inputs [None,dimension]
            y : desired outputs [None]

        Returns:
            If using SGD version, returns losses and rmse
        """
        if self.degree:
            X = self.poly.transform(X)
        self._initialized = True
        return self._fit(X, y)

    def _loss(self, y_true, y_pred):
        L = np.mean(self.loss_fn.loss(y_true, y_pred))
        Omega = 0
        if self.regularizer is not None:
            Omega = self.regularizer.value(self._weight)
        return L + Omega

    def _predict(self, X):
        return X @ self._weight + self._bias

    def predict(self, X):
        """compute output for given X

        Args:
            X : inputs [None,dimension]
        Raises:
            Exception: [description]

        Returns:
            returns outputs for X
        """
        if not self._initialized:
            raise Exception("Model hasn't fitted yet.")
        if self.degree is not None:
            X = self.poly.transform(X)

        return self._predict(X)

    def _rmse(self, y_true, y_pred):
        l = np.mean((y_pred - y_true) ** 2)
        return np.sqrt(l)

    def score(self, X, y):
        """Compute -rmse as score

        Args:
            X : inputs [None,dimension]
            y : desired outputs [None]

        Returns:
            return -rmse (score: higher better) 
        """
        y_pred = self.predict(X)
        return -self._rmse(y, y_pred)


class SGDRegressor(LinearRegressionBase):
    def __init__(
        self,
        learning_rate,
        epochs,
        loss_fn=MSELoss(),
        lambda_=None,
        clipnorm=None,
        batch_size=None,
        verbose=1,
        degree=None,
    ):
        """Stochastic Gradient Descent for linear regression 

        Args:
            learning_rate : learning_rate for using in stochastic gradient decsent
            epochs : numbers of epochs for training over each batch
            loss_fn (optional): Our loss for training our model. Defaults to MSELoss().
                                There's HuberLoss too.
            lambda_ (optional): Is hyperparameter for weighting our L2
                                regularization factor. Defaults to None.
            clipnorm (optional): Constrain on weights. Defaults to None.
            batch_size (optional): Size of mini batch. If not given it uses 
                                   whole batch Defaults to None.
            verbose (int, optional): If greater than zero it will print result on time.
                                     Defaults to 1.
            degree (optional): If set polynomial of features with interaction between them
                                will computed and will use. Defaults to None.
        """
        regularizer = None if lambda_ is None else L2(lambda_)
        super().__init__(loss_fn, regularizer, degree=degree)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.clipnorm = clipnorm
        self.verbose = verbose

    def _initialize(self, X):
        dim = np.shape(X)[1]
        self._weight = np.random.normal(
            loc=0.0, scale=1 / np.sqrt(dim + 1), size=(dim,)
        ).astype(X.dtype)
        self._bias = np.zeros((1,), dtype=X.dtype)

    def _batch_error(self, x_batch, y_batch):
        """Compute error for given batch

        Returns:
            bias error,weight_error
        """
        y_pred = self._predict(x_batch)
        error = self.loss_fn.derivative(y_batch, y_pred)

        bias_error = np.mean(error, axis=0)

        weight_error = error.reshape(-1, 1) * x_batch
        if self.regularizer is not None:
            weight_error += self.regularizer.derivative(self._weight)

        weight_error = np.mean(weight_error, axis=0)

        return bias_error, weight_error

    def _fit_batch(self, x_batch, y_batch):
        """Using given batch improve our bias,weights
        """
        bias_error, weight_error = self._batch_error(x_batch, y_batch,)
        if self.clipnorm:
            weight_norm = np.linalg.norm(weight_error)
            if weight_norm > self.clipnorm:
                weight_error = weight_error / weight_norm * self.clipnorm
                bias_error = bias_error / np.linalg.norm(bias_error) * self.clipnorm

        self._bias -= self.learning_rate * bias_error
        self._weight -= self.learning_rate * weight_error

    def _fit(self, X, y):
        """
        Train perceptron using X,y and prints out accuracy in
        process. Returns sequence of accuracies over
        each epoch.

        Args:
            X : Must be 2-D array (num_samples,num_dimentions)
            y : Label of X. Must be 2-D array (num_samples,)
            learning_rate : learning rate of perceptron algorithm
            epochs : number of training iteration
            lambda_ (optional): L2 regulazition term
            batch_size (optional): Number of sample for training
                    in each epoch. If set to 'None' algorithm use
                    all data for traninig. Defaults to None.

        Returns:
            1-D array (num_samples,):  sequence of losses over each epoch.
        """
        y = y.reshape((-1,))
        self._initialize(X)

        losses = np.empty(self.epochs + 1, dtype="float32")
        rmse = np.empty(self.epochs + 1, dtype="float32")

        y_pred = self._predict(X)
        rmse[0] = self._rmse(y, y_pred)
        losses[0] = self.loss_fn.loss(y, y_pred)
        del y_pred
        if self.verbose > 0:
            print(f"Epoch {0} => RMSE={rmse[0]} - Loss={losses[0]}", end="")

        for epoch in range(1, self.epochs + 1):
            if self.batch_size:
                x_batch, y_batch = self._get_batch(X, y, self.batch_size)
            else:
                x_batch, y_batch = X, y

            self._fit_batch(x_batch, y_batch)

            y_pred = self._predict(X)
            rmse[epoch] = self._rmse(y, y_pred)
            losses[epoch] = self.loss_fn.loss(y, y_pred)
            del y_pred
            if self.verbose > 0:
                print(
                    f"\rEpoch {epoch} => RMSE={rmse[epoch]} - Loss={losses[epoch]}",
                    end="",
                )
        print()
        return losses, rmse

    def _get_batch(self, X, y, batch_size):
        """Create mini batch of given size from (X,y)

        Returns:
            x_batch,y_batch
        """
        indices = np.random.randint(0, len(X), batch_size)
        return X[indices], y[indices]


class LinearRegression(LinearRegressionBase):
    def __init__(self, lambda_=None, degree=None):
        """Solve LinearRegression using NormalEquation

        Args:
            lambda_ (optional): Is hyperparameter for weighting our L2
                                regularization factor. Defaults to None.
            degree (optional): If set polynomial of features with interaction between them
                                will computed and will use. Defaults to None.
        """
        regularizer = None if lambda_ is None else L2(lambda_)
        self.lambda_ = lambda_
        super().__init__(MSELoss(), regularizer=regularizer, degree=degree)

    def _fit(self, X, y):
        X = self._add_ones_to_X(X)
        self._normal_equation(
            X, y,
        )

    def _add_ones_to_X(self, X):
        l, d = np.shape(X)
        A = np.empty((l, d + 1), dtype=X.dtype)
        A[:, 0] = 1
        A[:, 1:] = X
        return A

    def _pinv(self, X):
        if self.lambda_ is None:
            reg_term = 0
        else:
            reg_term = self.lambda_ * np.eye(np.shape(X)[1], dtype=X.dtype)
            reg_term[0, 0] = 0
        pinv = np.linalg.inv(X.T @ X + reg_term) @ X.T
        return pinv

    def _normal_equation(self, X, y):
        theta = self._pinv(X,) @ y
        self._bias = theta[0].reshape((-1,))
        self._weight = theta[1:].reshape((-1,))
