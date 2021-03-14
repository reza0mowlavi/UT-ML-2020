import numpy as np

from .initializer import random_initilizer
from .initializer import kmeanplus_initilizer

from joblib import Parallel
from joblib import delayed


class KMeans:
    def __init__(
        self,
        n_clusters,
        init="k-means++",
        n_init=10,
        tol=0.0001,
        n_iter=300,
        copy_x=False,
        n_jobs=None,
    ):
        """K-Means clustering.

        Args:
            n_clusters (int):The number of clusters to form as well as the number of centroids to generate.

            init (str, optional):{'k-means++', 'random'}, default='k-means++'. Method for initialization.

            n_init (int, optional):Number of time the k-means algorithm will be run with different
            centroid seeds. The final results will be the best output of n_init consecutive runs in
            terms of inertia. Defaults to 10.

            tol (float, optional):Relative tolerance with regards to inertia of model to declare convergence.
            If set to None then model doesn't consider it. Defaults to 0.0001.

            n_iter (int, optional):Maximum number of iterations of the k-means algorithm for a single run.
            Defaults to 300.

            n_jobs (int, optional):Number of parallel process for each run of LLoyd Algorithm.
            If set to -1 then it would use all cores. Defaults to None.

        Properties:
            assignments_ (array-like of shape (n_samples,)): Cluster assignment for each sample
            centroids_ (array-like of shape (n_clusters, n_features)) : Centroids for each cluster 
            epoch_ (int): Number of epochs to end of algorithm
            inertia_ (float): inertia for final result
            model_ : LLoyd model
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.tol = tol

    def fit(self, X, y=None):
        """Fit the model using X as training data and y as target values

        Args:
            X (array-like of shape (n_samples, n_features)): Training data
            y ([type], optional): Not needed and ignored

        Returns:
            object: self
        """
        tasks = [
            LLoydAlgorithm(
                n_clusters=self.n_clusters,
                init=self.init,
                tol=self.tol,
                n_iter=self.n_iter,
            )
            for _ in range(self.n_init)
        ]
        tasks = Parallel(n_jobs=self.n_jobs)(delayed(task.fit)(X) for task in tasks)

        inertias = [task.inertia_ for task in tasks]
        self.model_ = tasks[np.argmin(inertias)]

        self.assignments_ = self.model_.assignments_
        self.centroids_ = self.model_.centroids_
        self.inertia_ = self.model_.inertia_
        self.epochs_ = self.model_.epochs_
        self.converged_ = self.model_.converged_
        return self


class LLoydAlgorithm:
    def __init__(self, n_clusters=8, init="k-means++", tol=0.0001, n_iter=300):
        """LLoyd Algorithm

        Args:
             n_clusters (int):The number of clusters to form as well as the number of
            centroids to generate. Defaults to 8.

            init (str, optional):{'k-means++', 'random'}, default='k-means++'. Method for initialization.

            tol (float, optional):Relative tolerance with regards to inertia of model to declare convergence.
            If set to None then model doesn't consider it. Defaults to 0.0001.

            n_iter (int, optional):Maximum number of iterations of the k-means algorithm for a single run.
            Defaults to 300.

        Properties:
            assignments_ (array-like of shape (n_samples,)): Cluster assignment for each sample
            centroids_ (array-like of shape (n_clusters, n_features)) : Centroids for each cluster 
            epoch_ (int): Number of epochs to end of algorithm
            inertia_ (float): inertia for final result
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_iter = n_iter
        self.tol = tol

    def initialize(self, X):
        if self.init == "k-means++":
            initilizer = kmeanplus_initilizer
        elif self.init == "random":
            initilizer = random_initilizer
        else:
            raise ValueError('"init" parameter must be "random" or "k-means++".')

        self.centroids_ = np.copy(initilizer(X, self.n_clusters))
        self.assignments_ = np.zeros((np.shape(X)[0],), dtype="int32")

    def fit(self, X, y=None):
        """Fit the model using X as training data and y as target values

        Args:
            X (array-like of shape (n_samples, n_features)): Training data
            y ([type], optional): Not needed and ignored

        Returns:
            object: self
        """
        self.initialize(X)
        inertia = self.compute_inertia(X)
        self.converged_ = False
        for epoch in range(self.n_iter):
            self.centroid_assignment(X)
            self.centroid_moving(X)
            if self.tol is not None:
                new_inertia = self.compute_inertia(X)
                if abs(new_inertia - inertia) <= self.tol:
                    self.converged_ = True
                    break
                inertia = new_inertia

        self.inertia_ = self.compute_inertia(X)
        self.epochs_ = epoch + 1
        return self

    def centroid_assignment(self, X, y=None):
        distances = np.stack(
            [np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids_],
            axis=1,
        )
        self.assignments_ = np.argmin(distances, axis=1).astype("int32")

    def centroid_moving(self, X):
        self.centroids_ = np.array(
            [
                np.mean(X[self.assignments_ == centroid_num], axis=0)
                for centroid_num in range(np.shape(self.centroids_)[0])
            ]
        )

    def compute_inertia(self, X):
        inertia = np.mean(
            np.linalg.norm(X - self.centroids_[self.assignments_], axis=1)
        )
        return inertia
