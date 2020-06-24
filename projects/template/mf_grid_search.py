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

from simulation.configs import ExperimentConfig, set_model_config
from simulation.mf_experiments import (
    kfold_matrix_completion, 
    matrix_completion
)


def set_exp_config(hparams, counter, path_to_results):

    exp_config = ExperimentConfig(
        path_data_file="/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy",
        path_to_results=path_to_results,
        rank=int(hparams["rank"]),
        #num_train_samples=5,
        exp_id="run" + f"_param_combo{counter}", 
        save_only_configs=False,
        num_epochs=1000,
        n_kfold_splits=0,
        time_lag=4,
        epochs_per_display=1100,
        epochs_per_val=20,
        seed=42,
        monitor_loss=True,
        domain=[1, 4],
        early_stopping=True,
        val_size=0.2,
        patience=500,
        chances_to_improve=2
    )

    return exp_config


def run_grid_search():

    counter_offset = 

    #####   !!!
    #       !!!    TODO: Get weight matrix from analysis of unweighted model.
    ####    !!!

    model_type = "WMFTV"
    path_to_results = "/Users/sela/Desktop/tsd_code/results/wmf_tv/"
    
    experiment = "weighting"

    param_config = {
        "lambda1": [20],
        "lambda2": [0.01],
        "lambda3": [10],
        "init_basis": ["hmm"],
        "rank": [35],
        "num_iter": [70],
        "n_time_points": [321],
        "gamma": [0.5],
        "weighting": [""]
    }

    param_grid = ParameterGrid({**param_config})
    Parallel(n_jobs=)(
        delayed(matrix_completion)(
            set_exp_config(param_combo, counter + counter_offset, f"{path_to_results}/{experiment}"), 
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
