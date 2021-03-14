import numpy as np

from itertools import product, chain
from joblib import Parallel, delayed


def rmse_score(y_true, y_pred):
    l = np.mean((y_pred - y_true) ** 2)
    return -np.sqrt(l)


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))


class GridSearch:
    def __init__(
        self,
        clf,
        cv,
        param_grid,
        clf_params={},
        score_func=rmse_score,
        n_jobs=None,
        verbose=1,
    ):
        self.clf = clf
        self.cv = cv
        self.param_grid = param_grid
        self.score_func = score_func
        self.clf_params = clf_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def _fit(model, train_set, test_set, score_function):
        X_train, y_train = train_set
        X_test, y_test = test_set
        model.fit(X_train, y_train)
        score = score_function(y_test, model.predict(X_test))
        return score

    def fit(self, X, y):
        X, y = self._shuffle(X, y)
        data = self._get_batches(X, y)
        param_grid = dict_product(self.param_grid)
        best_mean_score = -np.inf
        best = None
        for param in param_grid:
            if self.verbose > 0:
                print(f"{param} -- in progress:", end="")
            tasks = (
                delayed(GridSearch._fit)(
                    self.clf(**self.clf_params, **param),
                    d["train"],
                    d["test"],
                    self.score_func,
                )
                for d in data
            )
            scores = Parallel(n_jobs=self.n_jobs, verbose=0)(tasks)
            mean_scores = np.mean(scores)
            if self.verbose > 0:
                print(f"\r{param} => mean:{mean_scores} -- std:{np.std(scores)}")

            if mean_scores > best_mean_score:
                best_mean_score = mean_scores
                best = {**param, "scores": scores, "best_mean_score": best_mean_score}

        self.best_ = best
        return best, best_mean_score

    def _get_batches(self, X, y):
        batch_size = int(np.ceil(len(X) / self.cv))
        data = []
        for num in range(self.cv):
            d = {}
            d["test"] = (
                X[num * batch_size : (num + 1) * batch_size],
                y[num * batch_size : (num + 1) * batch_size],
            )
            X_train = []
            y_train = []
            for i in chain(range(0, num,), range(num + 1, self.cv)):
                X_train.append(X[i * batch_size : (i + 1) * batch_size])
                y_train.append(y[i * batch_size : (i + 1) * batch_size])

            X_train = np.concatenate(X_train,)
            y_train = np.concatenate(y_train,)
            d["train"] = (X_train, y_train)
            data.append(d)
        return data

    def _shuffle(self, X, y):
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        return X, y
