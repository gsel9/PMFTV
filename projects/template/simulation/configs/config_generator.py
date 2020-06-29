import numpy as np
import tensorflow as tf

from simulation.configs import (
    WeightedMFConvConfig,
    WeightedMFTVConfig,
    MFConvConfig,
    MFLarsConfig, 
    MFTVConfig,
    GDLConfig
)

from ..utils.special_matrices import (
	laplacian_kernel_matrix,
	finite_difference_matrix
)


def set_model_config(hparams, model_type):

    if model_type == "GDL":

        # NOTE: Can have convergence criteria and skip optimising diffusion steps?
        return GDLConfig(
            row_graph_config=hparams["row_graph"], 
            col_graph_config=hparams["col_graph"],
            row_gamma=hparams["row_gamma"],
            col_gamma=hparams["col_gamma"],
            degree_row_poly=hparams["degree_row_poly"],
            degree_col_poly=hparams["degree_col_poly"],
            diffusion_steps=hparams["diffusion_steps"],
            channels=hparams["channels"],
            init_basis=hparams["init_basis"],
            init_coefs=hparams["init_coefs"],
            optimiser=hparams["optimiser"], 
            learning_rate=hparams["learning_rate"],
            loss=hparams["loss"]
        )

    if model_type == "MFConv":

        return MFConvConfig(
            lambda1=hparams["lambda1"],
            lambda2=hparams["lambda2"],
            lambda3=hparams["lambda3"],
            init_basis=hparams["init_basis"],
            K=laplacian_kernel_matrix(hparams["n_time_points"]),
            R=finite_difference_matrix(hparams["n_time_points"])
        )

    if model_type == "WMFConv":

        return WeightedMFConvConfig(
            lambda1=hparams["lambda1"],
            lambda2=hparams["lambda2"],
            lambda3=hparams["lambda3"],
            init_basis=hparams["init_basis"],
            weighting=hparams["weighting"],
            K=laplacian_kernel_matrix(hparams["n_time_points"]),
            R=finite_difference_matrix(hparams["n_time_points"])
        )

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
            R=finite_difference_matrix(hparams["n_time_points"]),
            J=np.ones((hparams["rank"], hparams["n_time_points"]))
        )

    if model_type == "WMFTV":

        return WeightedMFTVConfig(
            lambda1=hparams["lambda1"],
            lambda2=hparams["lambda2"],
            lambda3=hparams["lambda3"],
            gamma=hparams["gamma"],
            num_iter=hparams["num_iter"],
            init_basis=hparams["init_basis"],
            weighting=hparams["weighting"],
            R=finite_difference_matrix(hparams["n_time_points"]),
            J=np.ones((hparams["rank"], hparams["n_time_points"]))
        )

    raise ValueError(f"Unknown model type: {model_type}")
