from .linear_regression import LinearRegression
from .linear_regression import SGDRegressor
from .losses import MSELoss, HuberLoss
from .grid_search import GridSearch
from .polynomial_feature import PolyNomialFeatures
from .regularizer import L2


import numpy as np


def SSE(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


__all__ = [
    "LinearRegression",
    "SGDRegressor",
    "PolyNomialFeatures",
    "MSELoss",
    "HuberLoss",
    "L2",
    "SSE",
]
