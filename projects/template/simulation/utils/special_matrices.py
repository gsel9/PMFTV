import numpy as np


def finite_difference_matrix(T):

	return np.diag(np.pad(-np.ones(T - 1), (0, 1), 'constant')) + np.diag(np.ones(T-1), 1)


def laplacian_kernel_matrix(T):

	kernel = lambda x: np.exp(-1.0 * np.abs(x))

	return [kernel(np.arange(T) - i) for i in np.arange(T)]


def get_coefs(init_coefs, rank, X):

    if init_coefs == "random":
        np.random.seed(42)
        return np.random.random((X.shape[0], rank))


def get_basis(init_basis, rank, X):

    # NOTE: The initial approach. 
    if init_basis == "mean":

        return np.ones((X.shape[1], rank)) * np.mean(X[X.nonzero()])

    if init_basis == "svd":

        _, s, V = np.linalg.svd(X, full_matrices=False)
        return np.dot(U[:, :rank], np.diag(np.sqrt(s[:rank])))

    if init_basis == "random":
        np.random.seed(42)
        return np.random.choice(range(1, 5), size=(X.shape[1], rank), p=(0.7, 0.15, 0.1, 0.05))

    if init_basis == "hmm":

        np.random.seed(42)
        data = np.load("/Users/sela/Desktop/tsd_code/data/hmm/base_set_300K.npy")
        idx = np.random.choice(range(data.shape[0]), size=rank, replace=False)

        return np.transpose(data[idx])

    if init_basis == "smooth-hmm":

        np.random.seed(42)
        data = np.load("/Users/sela/Desktop/tsd_code/data/hmm/base_set_300K.npy")
        idx = np.random.choice(range(data.shape[0]), size=rank, replace=False)

        return np.transpose(data[idx])

    if init_basis == "noisy-logarithm":

        np.random.seed(42)
        # Half of profiles are logarithm functions and half are cancelling functions.
        # Add (standard) Gaussian noise for variability in the profiles.


def get_weight_matrix(weighting, X):

    if weighting == "identity":
        return np.eye(X.shape[0])

    if weighting == "max-state":
        return np.diag(np.max(X, axis=1))

    if weighting == "exp-max-state":
        # NOTE: Shift to not add extra weight to normals.
        return np.diag(np.exp(np.max(X, axis=1) - 1))

    if weighting == "exp-max-state-squared":
        # NOTE: Shift to not weight normals (weight coef = 1).
        w = np.max(X, axis=1) ** 2
        return np.diag(np.exp(w - 1))

    if weighting == "inv-imp":
        return np.diag(1 / np.max(X, axis=1))

    if weighting == "exp-inv-imp":
        return np.diag(1 / np.max(X, axis=1))

    # Current optimal.
    if weighting == "max-state-scaled":
        return np.diag(np.max(X, axis=1)) / 2

    if weighting == "max-state-scaled-max":
        return np.diag(np.max(X, axis=1)) / 4

    if weighting == "max-state-normed":
        W = np.diag(np.max(X, axis=1))
        return W / np.linalg.norm(W)

    if weighting == "binary":
        W_id = np.diag(np.max(X, axis=1))
        W = np.eye(X.shape[0])
        W[W_id > 2] = 2
        return W

    if weighting == "scaled-norm":
        W = np.linalg.norm(X, axis=1)
        return np.diag(W / max(W))

    if weighting == "sklearn-balanced":
        weights = class_weight.compute_class_weight('balanced', np.unique(X[X != 0]), X[X != 0])
        weights = weights / sum(weights)

        W = np.diag(np.max(X, axis=1))
        W[W == 1] = weights[0]
        W[W == 2] = weights[1]
        W[W == 3] = weights[2]
        W[W == 4] = weights[3]
        return W

    if weighting == "custom-balanced":
        weights = create_class_weight(X)

        W = np.diag(np.max(X, axis=1))
        W[W == 1] = weights[0]
        W[W == 2] = weights[1]
        W[W == 3] = weights[2]
        W[W == 4] = weights[3]
        return W

    if weighting == "normalised":
        return np.diag(np.max(X, axis=1)) / 4

    # Failed.
    if weighting == "log-max-state-scaled-max":
        w = 1 + np.log(np.max(X, axis=1))
        return np.diag(w / max(w))

    if weighting == "relative":
        m = np.max(X, axis=1)
        vals, counts = np.unique(m, return_counts=True)
        weights = {int(v): counts[np.argmax(counts)] / counts[i] for i, v in enumerate(vals)}
        w = np.zeros(X.shape[0])
        for c, weight in weights.items():
            w[m == int(c)] = weight

        return np.diag(w)
