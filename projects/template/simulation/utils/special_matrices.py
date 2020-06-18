import numpy as np


def finite_difference_matrix(T):

	return (np.diag(np.pad(-np.ones(T - 1), (0, 1), 'constant')) + np.diag(np.ones(T-1), 1))


def laplacian_kernel_matrix(T):

	kernel = lambda x: np.exp(-1.0 * np.abs(x))

	return [kernel(np.arange(T) - i) for i in np.arange(T)]


def init_basis(X_train, init_V, rank):

    if init_V == "ones":
        return np.ones((X_train.shape[1], rank)) * np.mean(X_train[X_train.nonzero()])
    
    if init_V == "svd":
        _, s, V = np.linalg.svd(X_train, full_matrices=False)
        return np.dot(np.diag(np.sqrt(s[:rank])), V[:rank, :]).T

    if init_V == "random":
        np.random.seed(42)
        return np.random.choice(range(1, 5), size=(X_train.shape[1], rank))

    if init_V == "random_float":
        np.random.seed(42)
        Z = np.random.random((X_train.shape[1], rank))
        return 1 + 3 * ((Z - np.min(Z)) / (np.max(Z) - np.min(Z)))

    if init_V == "random_skewed":
        np.random.seed(42)
        return np.random.choice(range(1, 5), size=(X_train.shape[1], rank), 
                                p=[0.78375552, 0.14345807, 0.0719589 , 0.00082751])

    if init_V == "random_skewed_ortho":
        np.random.seed(42)
        H = np.random.choice(range(1, 5), size=(X_train.shape[1], rank), 
                                p=[0.78375552, 0.14345807, 0.0719589 , 0.00082751])
        U, _, Vh = np.linalg.svd(H, full_matrices=False)
        return U @ Vh

    if init_V == "hmm":
        X_hmm = np.load("/Users/sela/Desktop/recsys_paper/data/hmm/40p/train/M_train.npy")
        np.random.seed(42)
        samples = np.random.choice(X_hmm.shape[0], size=rank)
        return X_hmm[samples].T

    if init_V == "hmm_svd":
        X_hmm = np.load("/Users/sela/Desktop/recsys_paper/data/hmm/40p/train/M_train.npy")
        np.random.seed(42)
        samples = np.random.choice(X_hmm.shape[0], size=rank)
        Z = X_hmm[samples]

        _, s, V = np.linalg.svd(Z, full_matrices=False)
        return np.dot(np.diag(np.sqrt(s[:rank])), V[:rank, :]).T

    if init_V == "ortho":
        np.random.seed(42)
        H = np.random.choice(range(1, 5), size=(X_train.shape[1], rank))
        U, _, Vh = np.linalg.svd(H, full_matrices=False)
        return U @ Vh
