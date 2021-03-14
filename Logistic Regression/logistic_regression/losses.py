import numpy as np


class BinaryCrossEntropy:
    def __init__(self, smothing=1e-7, class_weights=None):
        self.smoothing = smothing
        self.class_weights = class_weights

    def __call__(self, y_true, y_pred):
        class_weights = self.class_weights
        y_pred = np.clip(y_pred, self.smoothing, 1 - self.smoothing)
        l = -y_true * np.log(y_pred), -(1 - y_true) * np.log(1 - y_pred)
        if class_weights is not None:
            loss = l[0] * class_weights[0] + l[1] * class_weights[1]
        else:
            loss = l[0] + l[1]
        return loss

    def derivative(self, y_true, y_pred):
        class_weights = self.class_weights
        y_pred = np.clip(y_pred, self.smoothing, 1 - self.smoothing)
        d = -y_true / y_pred, (1 - y_true) / (1 - y_pred)
        if class_weights is not None:
            derivative = d[0] * class_weights[0] + d[1] * class_weights[1]
        else:
            derivative = d[0] + d[1]
        return derivative


class SparseCrossEntropy:
    def __init__(self, smothing=1e-7, class_weights=None):
        self.get = None
        if class_weights is not None:
            self.get = np.vectorize(class_weights.__getitem__)
        self.smoothing = smothing

    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.smoothing, 1 - self.smoothing)
        loss = -np.log(y_pred[np.arange(np.shape(y_true)[0]), y_true])
        if self.get is not None:
            loss = loss * self.get(y_true)
        return loss

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.smoothing, 1 - self.smoothing)

        derivative = np.array(
            [
                [
                    -int(y_true[k] == i) / y_pred[k, i]
                    for i in range(np.shape(y_pred)[1])
                ]
                for k in range(np.shape(y_true)[0])
            ],
            dtype=y_pred.dtype,
        )
        if self.get is not None:
            derivative = derivative * self.get(y_true)

        return derivative
