import numpy as np


def theta_mle(X, M):
    """Use reconstructed data to estimate the theta parameters used by the prediction
    algorithm.

    Args:
        X: Sparse data matrix
        M: Completed data matrix

    Returns:
        Theta estimate (float)
    """

    mask = (X != 0).astype(float)
    return np.sum(mask) / (2 * np.square(np.linalg.norm(mask * (X - M))))


def finite_difference_matrix(T):
    "Construct a (T x T) forward difference matrix"

    return np.diag(np.pad(-np.ones(T - 1), (0, 1), "constant")) + np.diag(
        np.ones(T - 1), 1
    )


def laplacian_kernel_matrix(T, gamma=1.0):
    "Construct a (T x T) matrix for convolutional regularization"

    def kernel(x):
        return np.exp(-1.0 * gamma * np.abs(x))

    return [kernel(np.arange(T) - i) for i in np.arange(T)]


def basis_baseline_value(shape, min_value):
    "Shifting the basic profiles so that min(V) >= min_value."
    return np.ones(shape) * min_value
