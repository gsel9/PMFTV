import numpy as np 

from sklearn.model_selection import ParameterGrid

from simulation.configs import MFConvConfig, ExperimentConfig
from simulation.mf_experiments import kfold_matrix_completion
from simulation.utils.special_matrices import (
    laplacian_kernel_matrix,
    finite_difference_matrix
)


def set_model_config(hparams):

    model_config = MFConvConfig(
        lambda1=hparams["lambda1"], 
        lambda2=hparams["lambda2"],
        lambda3=hparams["lambda3"],
        init_basis=hparams["init_basis"],
        J=np.zeros((hparams["n_time_points"], hparams["rank"])),
        K=laplacian_kernel_matrix(hparams["n_time_points"]),
        R=finite_difference_matrix(hparams["n_time_points"])
    )

    return model_config


def set_exp_config(hparams, counter):

    exp_config = ExperimentConfig(
        path_data_file="/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy",
        rank=hparams["rank"],
        num_train_samples=4000,
        exp_id="sanity_mfconv" + f"_param_combo{counter}", 
        path_to_results="/Users/sela/Desktop/tsd_code/results/mf_basis_init/",
        save_only_configs=True,
        num_epochs=1000,
        n_kfold_splits=2,
        time_lag=4,
        epochs_per_display=3000,
        epochs_per_val=2000,
        seed=42,
        domain=[1, 4],
        monitor_loss=False
    )

    return exp_config


def run_kfold_grid_search():

    counter = 0

    param_config = {
        "lambda1": [1],
        "lambda2": [1],
        "lambda3": [1, 3],
        "init_basis": ["hmm"],
        "rank": [15],
        "n_time_points": [321]
    }

    param_grid = ParameterGrid({**param_config})
    for param_combo in param_grid:

        kfold_matrix_completion(set_exp_config(param_combo, counter), set_model_config(param_combo))

        counter = counter + 1


if __name__ == '__main__':
    run_kfold_grid_search()
