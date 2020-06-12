import numpy as np 

from simulation.configs import MFLarsConfig, ExperimentConfig
from simulation.mf_experiments import kfold_matrix_completion
from simulation.utils.special_matrices import (
    laplacian_kernel_matrix,
    finite_difference_matrix
)


def run_kfold():

    rank = 15
    n_time_points = 321

    exp_config = ExperimentConfig(
        path_data_file="/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy",
        rank=rank,
        exp_id="kfold_run", 
        path_to_results="/Users/sela/Desktop/tsd_code/results/mf_rank/",
        save_only_configs=False,
        num_epochs=500,
        time_lag=4,
        n_kfold_splits=5,
        num_train_samples=4000,
        epochs_per_display=50,
        seed=42,
        domain=[1, 4]
    )

    model_config = MFLarsConfig(
        lambda2=0.001,
        lambda3=300,
        max_iter=max_iter,
        init_basis="hmm",
        J=np.zeros((n_time_points, rank)),
        K=laplacian_kernel_matrix(n_time_points),
        R=finite_difference_matrix(n_time_points)
    )

    kfold_matrix_completion(exp_config, model_config)
   

if __name__ == '__main__':
    run_kfold()
