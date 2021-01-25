import numpy as np


def finite_difference_matrix(T):

    return (np.diag(np.pad(-np.ones(T - 1), (0, 1), 'constant')) + np.diag(np.ones(T-1), 1))


def laplacian_kernel_matrix(T, gamma=1.0):

	kernel = lambda x: np.exp(-1.0 * gamma * np.abs(x))

	return [kernel(np.arange(T) - i) for i in np.arange(T)]