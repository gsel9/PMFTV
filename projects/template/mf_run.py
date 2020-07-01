import numpy as np 

from simulation.configs import ExperimentConfig, set_model_config
from simulation.data.dataset import TrainTestDataset
from simulation.mf_experiments import matrix_completion
from simulation.models.inference.map import MAP


def reconstruct_profiles():

    # MFConv; MFLars; MFTV; WMFConv; WMFTV; MFConv
    model_type = "WMFConv"
    base_path = "/Users/sela/Desktop/tsd_code/results"
    experiment = "wmf_conv/testing"
    exp_id = "init_run" 

    param_config = {
        "lambda0": 1.0,
        "lambda1": 1, 
        "lambda2": 1, 
        "lambda3": 300,
        "init_basis": "hmm",
        "rank": 25, 
        "n_time_points": 321,
        "weighting": "exp-max-state-squared"
    }

    # TODO:
    # * new filtered screening data
    # * Use input() to verify exp params + implement `set_exp_config`.
    # * Optimise `theta` as hyperparameter for prediction model.
    
    exp_config = ExperimentConfig(
        path_data_file="/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy",
        rank=param_config["rank"],
        exp_id=exp_id,
        path_to_results=f"{base_path}/{experiment}",
        save_only_configs=False,
        #num_train_samples=2,
        num_epochs=1000,
        time_lag=4,
        epochs_per_display=1100,
        seed=42,
        monitor_loss=True,
        domain=[1, 4],
        epochs_per_val=20,
        early_stopping=True,
        val_size=0.2,
        patience=200,
        chances_to_improve=2,
    )

    matrix_completion(exp_config, set_model_config(param_config, model_type=model_type))

    X_test = np.load("/Users/sela/Desktop/tsd_code/data/screening_filtered/test/X_test.npy")
    test_data = TrainTestDataset(X_test, time_lag=exp_config.time_lag)

    estimator = MAP(M_train=np.load(f"{base_path}/{experiment}/{exp_id}_M_hat.npy"), theta=2.5)
    y_pred = estimator.predict(test_data.X_train, test_data.time_of_prediction)

    np.save(f"{base_path}/{experiment}/{exp_id}_y_true.npy", test_data.y_true)
    np.save(f"{base_path}/{experiment}/{exp_id}_y_pred.npy", y_pred)
    

if __name__ == '__main__':
    reconstruct_profiles()
