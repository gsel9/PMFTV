import numpy as np

from simulation.configs import (
    WeightedMFConvConfig,
    WeightedMFTVConfig,
    MFLarsConfig, 
    MFTVConfig
)
from ..utils.special_matrices import (
	laplacian_kernel_matrix,
	finite_difference_matrix
)


def set_model_config(hparams, model_type):

    if model_type == "GDL":
        pass

    if model_type == "WMFConv":
        return WeightedMFConvConfig(
            lambda1=hparams["lambda1"],
            lambda2=hparams["lambda2"],
            lambda3=hparams["lambda3"],
            init_basis=hparams["init_basis"],
            W=None,
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
            R=finite_difference_matrix(hparams["n_time_points"])
        )

    if model_type == "WMFTV":

        return WeightedMFTVConfig(
            lambda1=hparams["lambda1"],
            lambda2=hparams["lambda2"],
            lambda3=hparams["lambda3"],
            gamma=hparams["gamma"],
            num_iter=hparams["num_iter"],
            init_basis=hparams["init_basis"],
            W=None,
            R=finite_difference_matrix(hparams["n_time_points"])
        )

    raise ValueError(f"Unknown model type: {model_type}")
