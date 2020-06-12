"""
mf_tv_param_config = {
    "lambda1": [1],
    "lambda2": [0.01],
    "lambda3": [10],
    "init_basis": ["hmm"],
    "rank": [10],
    "num_iter": [100], 
    "gamma": [0.5, 1, 1.5, 1.9], 
    "n_time_points": [321]
}
mf_lars_param_config = {
    "lambda2": [0.01],
    "lambda3": [10],
    "init_basis": ["hmm"],
    "rank": [30],
    "max_iter": [10, 15, 20, 25],
    "n_time_points": [321]
}
"""
import warnings

import numpy as np 

from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from simulation.configs import (
    MFLarsConfig, 
    MFTVConfig,
    ExperimentConfig
)
from simulation.mf_experiments import (
    kfold_matrix_completion, 
    matrix_completion
)
from simulation.utils.special_matrices import (
    laplacian_kernel_matrix,
    finite_difference_matrix
)


def set_model_config(hparams, model_type):

    if model_type == "MFLars":
        
        return MFLarsConfig(
            lambda2=hparams["lambda2"],
            lambda3=hparams["lambda3"],
            max_iter=hparams["max_iter"],
            init_basis=hparams["init_basis"],
            J=np.ones((hparams["n_time_points"], hparams["rank"])),
            #np.zeros((hparams["n_time_points"], hparams["rank"])),
            K=laplacian_kernel_matrix(hparams["n_time_points"]),
            R=finite_difference_matrix(hparams["n_time_points"])
        )

    if model_type == "MFTV":

        return MFTVConfig(
            lambda1=hparams["lambda1"],
            lambda2=hparams["lambda2"],
            lambda3=hparams["lambda3"],
            gamma=hparams["gamma"],
            num_iter=hparams["num_iter"],
            init_basis=hparams["init_basis"],
            J=np.ones((hparams["n_time_points"], hparams["rank"])),
            #np.zeros((hparams["n_time_points"], hparams["rank"])),
            R=finite_difference_matrix(hparams["n_time_points"])
        )

    raise ValueError(f"Invalid model type: {model_type}")


def set_exp_config(hparams, counter, path_to_results):

    exp_config = ExperimentConfig(
        path_data_file="/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy",
        path_to_results=path_to_results,
        rank=int(hparams["rank"]),
        num_train_samples=6000,
        exp_id="run" + f"_param_combo{counter}", 
        save_only_configs=False,
        num_epochs=500,
        n_kfold_splits=2,
        time_lag=4,
        epochs_per_display=1100,
        epochs_per_val=1100,
        seed=42,
        monitor_loss=True,
        domain=[1, 4],
        early_stopping=False,
        #shuffle=False,
        val_size=0.2,
        #patience=1
    )

    return exp_config


def run_grid_search():

    counter_offset = 0

    model_type = "MFLars"
    path_to_results = "/Users/sela/Desktop/tsd_code/results/mf_lars/greedy_opt"

    param_config = {
        "lambda2": [0.1, 1],
        "lambda3": [0.01, 0.1, 1],
        "init_basis": ["hmm"],
        "rank": [50],
        "max_iter": [24],
        "n_time_points": [321]
    }

        """
    counter_offset = 28

    model_type = "MFTV"
    path_to_results = "/Users/sela/Desktop/tsd_code/results/mf_tv/greedy_opt"
    param_config = {
        "lambda1": [1],
        "lambda2": [0.001, 5],
        "lambda3": [8],
        "init_basis": ["hmm"],
        "rank": [24],
        "num_iter": [70], 
        "gamma": [0.5], 
        "n_time_points": [321]
    }
    """

    param_grid = ParameterGrid({**param_config})
    Parallel(n_jobs=3)(
        delayed(matrix_completion)(
            set_exp_config(param_combo, counter + counter_offset, path_to_results), 
            set_model_config(param_combo, model_type=model_type)
        ) 
        for counter, param_combo in enumerate(param_grid)
    )


def run_kfold_grid_search():

    counter_offset = 0

    param_config = {
        "max_iter": [1],
        "lambda2": [1],
        "lambda3": [1, 3],
        "init_basis": ["hmm"],
        "rank": [15],
        "n_time_points": [321]
    }

    param_grid = ParameterGrid({**param_config})
    for param_combo in param_grid:
        kfold_matrix_completion(set_exp_config(param_combo, counter + counter_offset), set_model_config(param_combo))
        counter = counter + 1
   

if __name__ == '__main__':
    #run_kfold_grid_search()
    run_grid_search()
