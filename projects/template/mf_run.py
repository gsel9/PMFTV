import numpy as np 

from simulation.configs import ExperimentConfig, set_model_config
from simulation.data.dataset import TrainTestDataset
from simulation.mf_experiments import matrix_completion
from simulation.models.inference.map import MAP


def reconstruct_profiles():

    # MFConv; MFLars; MFTV; WMFConv: WMFTV
    model_type = "WMFConv"
    base_path = "/Users/sela/Desktop/tsd_code/results"
    experiment = "wmf_conv/testing"
    exp_id = "init9"

    param_config = {
        "lambda1": 0.5,
        "lambda2": 0.001,
        "lambda3": 300,
        "init_basis": "hmm",
        "rank": 20, 
        "n_time_points": 321
    }

    # TODO: 
    # * new filtered screening data
    # * Use input() to verify exp params + implement `set_exp_config`.
    # * Optimise `theta` as hyperparameter for prediction model.

    exp_config = ExperimentConfig(
        path_data_file="/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy",
        #path_data_file=f"/Users/sela/Desktop/recsys_paper/data/dgd/{exp_id}/train/X_train.npy",
        rank=param_config["rank"],
        exp_id=exp_id,
        path_to_results=f"{base_path}/{experiment}",
        save_only_configs=False,
        #num_train_samples=5, #1000,
        num_epochs=1000,
        time_lag=4,
        epochs_per_display=100,
        epochs_per_val=20,
        seed=42,
        domain=[1, 4],
        #early_stopping=False,
        #val_size=0.2,
        #patience=150
    )

    matrix_completion(exp_config, set_model_config(param_config, model_type=model_type))

    X_test = np.load("/Users/sela/Desktop/tsd_code/data/screening_filtered/test/X_test.npy")
    #X_test = np.load(f"/Users/sela/Desktop/recsys_paper/data/dgd/{exp_id}/test/X_test.npy")
    test_data = TrainTestDataset(X_test, time_lag=exp_config.time_lag)

    estimator = MAP(M_train=np.load(f"{base_path}/{experiment}/{exp_id}_M_hat.npy"), theta=2.5)
    y_pred = estimator.predict(test_data.X_train, test_data.time_of_prediction)

    np.save(f"{base_path}/{experiment}/{exp_id}_y_true.npy", test_data.y_true)
    np.save(f"{base_path}/{experiment}/{exp_id}_y_pred.npy", y_pred)


if __name__ == '__main__':
    reconstruct_profiles()
