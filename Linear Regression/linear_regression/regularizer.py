import numpy as np


class L2:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def value(self, weights):
        return 0.5 * self.lambda_ * (weights @ weights)

    def derivative(self, weights):
        return weights * self.lambda_
