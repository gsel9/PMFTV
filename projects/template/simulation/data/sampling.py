import numpy as np 

from .dataset import TrainTestDataset


OVERSAMPLING = {
    1: lambda X: int(X.shape[0] * 1),
    2: lambda X: int(X.shape[0] * 1),
    3: lambda X: int(X.shape[0] * 1),
    4: lambda X: int(X.shape[0] * 1)
}


UNDERSAMPLING = {
    1: 72,
    2: 72,
    3: 72,
    4: 72,
}


STRATIFIED_SAMPLING = {
    1: lambda X: int(X.shape[0] * 5),
    2: lambda X: int(X.shape[0] * 0.5),
    3: lambda X: int(X.shape[0] * 0.25),
    4: lambda X: int(X.shape[0] * 0.25)
}


def stratified_resample(X, seed=42, shuffle=True):
    # Resampling histories for a more balanced dataset.

    np.random.seed(seed)

    m = np.max(X, axis=1)
    idx = np.arange(len(m))

    X_new = []
    for c in np.unique(m):
        c_idx = np.random.choice(idx[m == c], replace=True, size=UNDERSAMPLING[c])
            #size=STRATIFIED_SAMPLING[c](X))
        X_new.extend(X[c_idx])

    np.random.shuffle(X_new)
        
    return  np.array(X_new)


def resample(X, seed=42, shuffle=True):
    # Resampling histories for a more balanced dataset.

    # TEMP:
    return stratified_resample(X)

    """
    np.random.seed(seed)

    m = np.max(X, axis=1)
    n = X.shape[0] // len(np.unique(m))

    idx = np.arange(len(m))

    X_new = []
    for c in np.unique(m):
        c_idx = np.random.choice(idx[m == c], replace=True, size=n)
        X_new.extend(X[c_idx])

    np.random.shuffle(X_new)
        
    return  np.array(X_new)
    """


def sample_subset(X, num_subset_samples, seed, return_index=False):

    np.random.seed(seed)
    idx = np.random.choice(range(X.shape[0]), replace=False, size=num_subset_samples)

    print(f"Selected subset of {num_subset_samples} samples")
    if return_index:
    	return idx, X[idx]

    return X[idx]


def sample_subgroup(X, subgroup, missing=0):
    # Select only the histories with the diagnoses in `subgroup`.

    X_sub = X.copy()

    m = np.max(X, axis=1)
    if isinstance(subgroup, int):
        return X_sub[m <= subgroup]

    if isinstance(subgroup, list):
        return X_sub[np.logical_and(m >= subgroup[0], m <= subgroup[1])]


def sample_validation_set(X, exp_config):

    X_val = sample_subset(X, exp_config.num_subset_samples, exp_config.seed)

    val_set = TrainTestDataset(X_val, time_lag=exp_config.time_lag)
