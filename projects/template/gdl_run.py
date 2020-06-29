import numpy as np 

from simulation.configs import ExperimentConfig, set_model_config
from simulation.data.dataset import TrainTestDataset
from simulation.gdl_experiments import matrix_completion
from simulation.models.inference.map import MAP


def reconstruct_profiles():

    model_type = "GDL"
    base_path = "/Users/sela/Desktop/tsd_code/results"
    experiment = "gdl/testing"
    exp_id = "undersampling" 

    # NB: Make sure all metadata is saved to disk.
    param_config = {
        "rank": 15,
        "row_gamma": 0.001,
        "col_gamma": 0.001, 
        "degree_row_poly": 5,
        "degree_col_poly": 5,
        "diffusion_steps": 10,
        "channels": 32,
        "optimiser": "Adam",
        "loss": "original",
        "learning_rate": 1e-3,
        "init_coefs": "random",
        "init_basis": "hmm",
        "row_graph": {
            "path_distance_matrix": "/Users/sela/Desktop/tsd_code/data/dgd/40p/train/wasserstein_p1.npy",
            "method": "knn",
            "k": 10
        },
        "col_graph": {
            "method": "sequential",
            "num_nodes": 321,
            "weights": [i for i in np.exp(-np.arange(2))]
        }
    }
    
    exp_config = ExperimentConfig(
        path_data_file="/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy",
        rank=param_config["rank"],
        exp_id=exp_id,
        path_to_results=f"{base_path}/{experiment}",
        save_only_configs=False,
        num_train_samples=10,
        num_epochs=500,
        n_kfold_splits=10,
        time_lag=4,
        epochs_per_display=1100,
        seed=42,
        monitor_loss=True,
        domain=[1, 4],
        resample=True,
        early_stopping=False,
        val_size=0,
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
