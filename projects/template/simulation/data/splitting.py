import numpy as np 

from sklearn.model_selection import train_test_split

from .dataset import TrainTestDataset


def train_val_split(X, exp_config):

    train_idx, val_idx = train_test_split(range(X.shape[0]), 
    									  test_size=exp_config.val_size, 
    									  random_state=exp_config.seed, 
    									  shuffle=exp_config.shuffle)

    print(f"Splitted into {X[train_idx].shape} training and "
          f"{X[val_idx].shape} validation sets")

    exp_config.update_value("num_train_samples", len(train_idx))
    exp_config.update_value("num_val_samples", len(val_idx))

    val_set = TrainTestDataset(X[val_idx], time_lag=exp_config.time_lag)

    return X[train_idx], val_set


def sample_subset(X, num_subset_samples, seed, return_index=False):

    np.random.seed(seed)
    idx = np.random.choice(range(X.shape[0]), replace=False, size=num_subset_samples)

    print(f"Selected subset of {num_subset_samples} samples")
    if return_index:
    	return idx, X[idx]

    return X[idx]


def sample_validation_set(X, exp_config):

    X_val = sample_subset(X, exp_config.num_subset_samples, exp_config.seed)

    val_set = TrainTestDataset(X_val, time_lag=exp_config.time_lag)
