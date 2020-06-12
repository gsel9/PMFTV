import numpy as np 

from simulation.configs import MFConvConfig, ExperimentConfig
from simulation.mf_experiments import matrix_completion

from simulation.data.dataset import TrainTestDataset
from simulation.utils.special_matrices import (
    laplacian_kernel_matrix,
    finite_difference_matrix
)
from simulation.models.inference.map import MAP


def reconstruct_profiles():

    rank = 15
    n_time_points = 321

    exp_config = ExperimentConfig(
        path_data_file="/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy",
        rank=rank,
        exp_id="hmm_smooth", 
        path_to_results="/Users/sela/Desktop/tsd_code/results/mf_basis_init/",
        save_only_configs=False,
        num_epochs=1000,
        time_lag=4,
        epochs_per_display=1000,
        epochs_per_val=50,
        seed=42,
        domain=[1, 4],
        early_stopping=True,
        shuffle=False,
        val_size=0.2,
        patience=5
    )

    model_config = MFConvConfig(
        lambda1=0.5, 
        lambda2=0.001,
        lambda3=300,
        init_basis="hmm_smooth",
        J=np.zeros((n_time_points, rank)),
        K=laplacian_kernel_matrix(n_time_points),
        R=finite_difference_matrix(n_time_points)
    )

    matrix_completion(exp_config, model_config)


def map_estimate():

    X_test = np.load("/Users/sela/Desktop/tsd_code/data/screening_filtered/test/X_test.npy")
    test_data = TrainTestDataset(X_test, time_lag=4, random_state=42)

    M_hat = np.load("/Users/sela/Desktop/tsd_code/results/sanity/sanity_mfconv_M_hat.npy")
    estimator = MAP(M_train=M_hat, theta=2.5)

    y_pred = estimator.predict(test_data.X_train, test_data.time_of_prediction)

    np.save("/Users/sela/Desktop/tsd_code/results/sanity/y_true.npy", test_data.y_true)
    np.save("/Users/sela/Desktop/tsd_code/results/sanity/y_pred.npy", y_pred)


if __name__ == '__main__':
    reconstruct_profiles()
    #map_estimate()
