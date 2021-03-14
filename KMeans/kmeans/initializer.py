import numpy as np


def kmeanplus_initilizer(X, n_clusters):
    """KMeans++ initializer for KMeans

    Args:
        X (array-like of shape (n_samples, n_features)): Training data
        n_clusters (int):The number of centroids to generate.

    Returns:
        array-like of shape (n_clusters, n_features): ndarray of centroids
    """
    N = np.shape(X)[0]
    xs = range(N)
    centroids = np.empty((n_clusters, np.shape(X)[1]), dtype=X.dtype)
    centroids[0] = X[np.random.randint(0, N)]

    for i in range(1, n_clusters):
        distances = np.stack(
            [np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]],
            axis=1,
        )
        assignments = np.argmin(distances, axis=1).astype("int32")
        distances = np.linalg.norm(X - centroids[assignments], axis=1)
        S = np.sum(distances)
        probs = distances / S
        centroids[i] = X[np.random.choice(xs, 1, p=probs)]

    return centroids


def random_initilizer(X, n_clusters):
    """Random initializer for KMeans

    Args:
        X (array-like of shape (n_samples, n_features)): Training data
        n_clusters (int):The number of centroids to generate.

    Returns:
        array-like of shape (n_clusters, n_features): ndarray of centroids
    """
    indices = np.arange(np.shape(X)[0])
    indices = np.random.choice(indices, n_clusters, replace=False)
    return X[indices]

