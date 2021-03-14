import numpy as np


class HuberLoss:
    def __init__(self, delta=1):
        self.delta = delta

    def loss(self, y_true, y_pred):
        delta = self.delta
        abs_error = np.abs(y_true, y_pred)
        abs_error = np.where(
            abs_error <= delta,
            np.square(abs_error) * 0.05,
            delta * abs_error - 0.5 * np.square(delta),
        )
        return np.mean(abs_error)

    def derivative(self, y_true, y_pred):
        delta = self.delta
        error = y_true - y_pred
        abs_error = np.abs(error)

        indices = abs_error <= delta
        abs_error[indices] = -error[indices]

        not_indices = np.logical_not(indices)
        del indices
        abs_error[not_indices] = np.where(error[not_indices] >= 0, -1, 1)
        return abs_error


class MSELoss:
    def loss(self, y_true, y_pred):
        return np.mean(0.5 * np.square(y_pred - y_true))

    def derivative(self, y_true, y_pred):
        return y_pred - y_true
