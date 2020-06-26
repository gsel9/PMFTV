import numpy as np

from .sampling import sample_subset, sample_subgroup, resample


def data_summary(data, run_config=None, training=True):

    v, c = np.unique(data[data != 0], return_counts=True)
    print("Values {}; Counts {}".format(v, c))

    if run_config is not None:

        if training:
            run_config.update_config({"unique_train_values": v, "train_distribution": c})
        else:
            run_config.update_config({"unique_val_values": v, "val_distribution": c})


def load_data_matrix(exp_config, training=True):

    X = np.load(exp_config.path_data_file)
    print(f"Loaded {np.shape(X)} data matrix from: {exp_config.path_data_file}")

    # NOTE: exp_config.subgroup can be <int> or <list>.
    if exp_config.subgroup is not None:
        print("Sampling subset")
        X = sample_subgroup(X, exp_config.subgroup)

    if exp_config.resample:
        print("Resampling")
        X = resample(X)

    if training:
        num_subset_samples = exp_config.num_train_samples
    else:
        num_subset_samples = exp_config.num_val_samples

    idx = None
    if num_subset_samples is not None:
        idx, X = sample_subset(X, num_subset_samples, exp_config.seed, return_index=True)

    return idx, X


def load_data_from_file(exp_config, run_config=None):

    idx, X = load_data_matrix(exp_config) 
    data_summary(data=X, run_config=run_config)

    return idx, X
