import numpy as np


class Sigmiod:
    def __init__(self, MAX_EXP=709):
        self.MAX_EXP = 709

    def __call__(self, Z):
        Z = np.clip(Z, -self.MAX_EXP, self.MAX_EXP)
        return 1 / (1 + np.exp(-Z))

    def derivative(self, Z):
        Z = np.clip(Z, -self.MAX_EXP, self.MAX_EXP)
        exp = np.exp(-Z)
        return exp / (1 + exp) ** 2


class Softmax:
    def __call__(self, Z, trainig=True):
        max_val = np.max(Z, axis=1, keepdims=True)
        Z = Z - max_val
        numerator = np.exp(Z)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        A = numerator / denominator
        self.training = trainig
        if trainig:
            self.A = A
        return A

    def derivative(self, Z=None, return_jacobian=False):
        A = self.A if Z is None else self(Z)
        N, D = np.shape(A)

        """ 
        derivative = np.empty((N, D, D), dtype=A.dtype)
        for k in range(N):
            for i in range(D):
                for j in range(D):
                    if i == j:
                        derivative[k, i, j] = -A[k, i] * A[k, j]
                    else:
                        derivative[k, i, j] = A[k, i] * (1 - A[k, i]) """
        derivative = np.array(
            [
                [
                    [A[k, i] * (int(i == j) - A[k, j]) for j in range(D)]
                    for i in range(D)
                ]
                for k in range(N)
            ],
            dtype=A.dtype,
        )

        self.A = None
        return derivative

