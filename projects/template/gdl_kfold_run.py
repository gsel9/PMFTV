import numpy as np

from simulation.configs import GDLConfig, RowGraphConfig, ColumnGraphConfig, ExperimentConfig
from simulation.gdl_matrix_completion import matrix_completion

from sklearn.model_selection import ParameterGrid


def main(exp_params, model_params):

    exp_config = ExperimentConfig(
        path_to_datadir="../../tmp_data",
        rank=5,
        num_train_samples=100,
        num_val_samples=100,
        exp_id="hei", 
        path_to_results="../../tmp_results",
        save_only_configs=None,
        num_epochs=1000,
        prediction_window=4,
        epochs_per_display=3,
        epochs_per_val=2,
        seed=42,
        domain=[1, 4]
    )

    model_config = GDLConfig(
        name="GDL",
        row_gamma=0.001,
        col_gamma=0.001, 
        tv_gamma=None,
        conv_gamma=None, 
        degree_row_poly=4, 
        degree_col_poly=4, 
        diffusion_steps=4,
        channels=10,
        optimiser="adam",
        learning_rate=1e-3
    )

    row_graph_config = RowGraphConfig(
        path_distance_matrix="../../tmp_data/wasserstein_p1.npy",
        method="knn",
        k=4
    )

    col_graph_config = ColumnGraphConfig(
        method="sequential",
        num_nodes=321,
        weights=[i for i in np.exp(-np.arange(5))]
    )

    matrix_completion(exp_config, model_config, row_graph_config, col_graph_config)


def main():

	exp_params  ={
		"path_to_datadir": "../../tmp_data",
        "rank": 5,
        "num_train_samples": 100,
        "num_val_samples": 100,
        "exp_id": "hei", 
        "path_to_results": "../../tmp_results",
        "save_only_configs": None,
        "num_epochs": 2,
        "prediction_window": 4,
        "epochs_per_display": 3,
        "epochs_per_val": 2,
        "seed": 42,
        "cv_splits": 5,
        "domain": [1, 4]
	}

	screening_grid = {
		"row_gamma": [0.001],
        "col_gamma": [0.001],
        "tv_gamma": [0.05],
        "degree_row_poly": [5],
        "degree_col_poly": [5],
        "diffusion_steps": [10], 
        "learning_rate": [1e-3],
        "channels": [30],
        "k": [5],
        "rank": [5]
    }

	param_grid = ParameterGrid({**param_config})

	counter = 0	
	for model_params in param_grid:

		exp_params["exp_id"] = f"{exp_params["exp_id"]}_combo{counter}"

		run_param_config()

		counter = counter + 2


if __name__ == '__main__':
	main()
