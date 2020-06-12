import numpy as np

from simulation.configs import GDLConfig, RowGraphConfig, ColumnGraphConfig, ExperimentConfig
from simulation.gdl_matrix_completion import matrix_completion


def main():

    exp_config = ExperimentConfig(
        path_to_datadir="/Users/sela/Desktop/tsd_code/data/dgd/40p",
        rank=5,
        num_train_samples=500,
        num_val_samples=500,
        exp_id="conv", 
        path_to_results="/Users/sela/Desktop/tsd_code/tmp_results",
        save_only_configs=False,
        num_epochs=300,
        prediction_window=4,
        epochs_per_display=100,
        epochs_per_val=100,
        seed=42,
        domain=[1, 4]
    )

    model_config = GDLConfig(
        name="GDL",
        row_gamma=0.001,
        col_gamma=0.001, 
        tv_gamma=None,
        conv_gamma=None, 
        degree_row_poly=5,
        degree_col_poly=5,
        diffusion_steps=10,
        channels=32,
        optimiser="adam",
        learning_rate=1e-3
    )

    row_graph_config = RowGraphConfig(
        path_distance_matrix="/Users/sela/Desktop/tsd_code/data/dgd/40p/train/wasserstein_p1.npy",
        method="knn",
        k=10
    )

    col_graph_config = ColumnGraphConfig(
        method="sequential",
        num_nodes=321,
        weights=[i for i in np.exp(-np.arange(5))]
    )

    matrix_completion(exp_config, model_config, row_graph_config, col_graph_config)


if __name__ == '__main__':
    main()
