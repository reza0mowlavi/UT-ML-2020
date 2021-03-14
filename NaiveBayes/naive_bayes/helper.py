import numpy as np


class StratifiedShuffleSplit:
    def __init__(self, test_size, shuffle=True):
        self.test_size = test_size
        self.shuffle = shuffle

    def split(self, X, y):
        if self.shuffle:
            X, y = self._shuffle_data(X, y)

        N = np.shape(y)[0]

        test_size = self.test_size * N if self.test_size < 1 else self.test_size

        data, labels = self._get_dataset(X, y)
        fracs = [np.shape(d)[0] for d in data]
        fracs = np.divide(fracs, np.sum(fracs))
        test_sizes = np.floor(np.multiply(fracs, test_size)).astype("int64")
        X_trains, y_trains = [], []
        X_tests, y_tests = [], []
        for d, label, test_size in zip(data, labels, test_sizes):
            X_test, X_train = d[:test_size], d[test_size:]

            y_test = np.empty(test_size,)
            y_test.fill(label)

            y_train = np.empty(np.shape(d)[0] - test_size,)
            y_train.fill(label)

            X_trains.append(X_train)
            y_trains.append(y_train)

            X_tests.append(X_test)
            y_tests.append(y_test)

        del data, labels

        X_train, y_train = (
            np.concatenate(X_trains, axis=0),
            np.concatenate(y_trains, axis=0),
        )
        del X_trains, y_trains
        X_test, y_test = (
            np.concatenate(X_tests, axis=0),
            np.concatenate(y_tests, axis=0),
        )
        del X_tests, y_tests
        if self.shuffle:
            X_train, y_train = self._shuffle_data(X_train, y_train)
            X_test, y_test = self._shuffle_data(X_test, y_test)
        return (X_train, y_train), (X_test, y_test)

    def _shuffle_data(self, X, y):
        indices = np.arange(np.shape(X)[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def _get_dataset(self, X, y):
        data = []
        labels = np.unique(y)
        for label in labels:
            data.append(X[y == label])
        return data, labels.astype("int64")


class OrdinalEncoderColumn:
    def fit(self, features):
        unique_vals = np.unique(features)
        self.mapping = {unique_val: i + 1 for i, unique_val in enumerate(unique_vals)}
        get = lambda x: self.mapping.get(x, 0)
        self.get = np.vectorize(get)
        return self

    def transform(self, features):
        return self.get(features)


class StandardScaler:
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        self.means = np.mean(X, axis=0, keepdims=True)
        self.var = np.var(X, axis=0, keepdims=True)
        return self

    def transform(self, X):
        return (X - self.means) / np.sqrt(self.var + self.epsilon)
