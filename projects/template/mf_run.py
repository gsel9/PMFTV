import numpy as np 

from simulation.configs import (
    MFLarsConfig, 
    MFTVConfig,
    ExperimentConfig
)
from simulation.mf_experiments import matrix_completion

from simulation.data.dataset import TrainTestDataset
from simulation.utils.special_matrices import (
    laplacian_kernel_matrix,
    finite_difference_matrix
)
from simulation.models.inference.map import MAP


def set_model_config(hparams, model_type):

    if model_type == "MFLars":
        
        return MFLarsConfig(
            lambda2=hparams["lambda2"],
            lambda3=hparams["lambda3"],
            max_iter=hparams["max_iter"],
            init_basis=hparams["init_basis"],
            J=np.zeros((hparams["n_time_points"], hparams["rank"])),
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


def reconstruct_profiles():

    # MFConv; MFLars; MFTV
    model_type = "MFTV"
    exp_id = "init1"
    experiment = "testing"
    base_path = "/Users/sela/Desktop/tsd_code/results/mf_tv"

    param_config = {
        "lambda1": 1,
        "lambda2": 0.01,
        "lambda3": 8,
        "init_basis": "hmm",
        "rank": 24,
        "num_iter": 70,
        "gamma": 0.5,
        "n_time_points": 321
    }

    exp_config = ExperimentConfig(
        path_data_file="/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy",
        rank=param_config["rank"],
        exp_id=exp_id, 
        path_to_results=f"{base_path}/{experiment}",
        save_only_configs=False,
        num_epochs=2000,
        time_lag=4,
        #num_train_samples=50,
        epochs_per_display=100,
        #epochs_per_val=50,
        seed=42,
        domain=[1, 4],
        #early_stopping=False,
        #shuffle=False,
        #val_size=0.2,
        #patience=2
    )

    matrix_completion(exp_config, set_model_config(param_config, model_type=model_type))

    X_test = np.load("/Users/sela/Desktop/tsd_code/data/screening_filtered/test/X_test.npy")
    test_data = TrainTestDataset(X_test, time_lag=exp_config.time_lag)

    M_hat = np.load(f"{base_path}/{experiment}/{exp_id}_M_hat.npy")
    estimator = MAP(M_train=M_hat, theta=2.5)

    y_pred = estimator.predict(test_data.X_train, test_data.time_of_prediction)

    np.save(f"{base_path}/{experiment}/{exp_id}_y_true.npy", test_data.y_true)
    np.save(f"{base_path}/{experiment}/{exp_id}_y_pred.npy", y_pred)


if __name__ == '__main__':
    reconstruct_profiles()
