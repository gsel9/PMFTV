import warnings

import numpy as np

from sklearn.utils import class_weight

from .gdl import MGCNN
from .gdl.graphs import row_laplacian, column_laplacian

from .matrix_factorisation import (
    MFConv, MFLars, MFTV, WeightedMFConv, WeightedMFTV
)

from ..utils.special_matrices import get_weight_matrix


def get_gdl_model(X, exp_config, model_config, subset_idx=None):

    print("Instantiating GDL model:")
    print(f"* channels: {model_config.channels}")
    print(f"* degree_row_poly: {model_config.degree_row_poly}")
    print(f"* degree_col_poly: {model_config.degree_col_poly}")
    print(f"* diffusion_steps: {model_config.diffusion_steps}")
    print(f"* rank: {exp_config.rank}")

    model =  MGCNN(
        X_train=X,
        Lr=row_laplacian(model_config.row_graph_config, subset_idx),
        Lc=column_laplacian(model_config.col_graph_config),
        rank=exp_config.rank,
        domain=exp_config.domain,
        n_conv_feat=model_config.channels, 
        ord_row_conv=model_config.degree_row_poly, 
        ord_col_conv=model_config.degree_col_poly,
        diffusion_steps=model_config.diffusion_steps
    )

    return model


def get_weighted_mf_conv_model(X, exp_config, model_config):

    print("Instantiating Weighted Convolutional MF model:")
    print(f"* lambda1: {model_config.lambda1}")
    print(f"* lambda2: {model_config.lambda2}")
    print(f"* lambda3: {model_config.lambda3}")
    print(f"* rank: {exp_config.rank}")

    model = WeightedMFConv(
        X_train=X, 
        V_init=get_basis(model_config.init_basis, rank=exp_config.rank, X=X), 
        R=model_config.R, 
        W=get_weight_matrix(model_config.weighting, X=X),
        K=model_config.K,
        lambda1=model_config.lambda1, 
        lambda2=model_config.lambda2, 
        lambda3=model_config.lambda3,
        rank=exp_config.rank
    )

    return model


def get_mf_conv_model(X, exp_config, model_config):

    print("Instantiating Convolutional MF model:")
    print(f"* lambda1: {model_config.lambda1}")
    print(f"* lambda2: {model_config.lambda2}")
    print(f"* lambda3: {model_config.lambda3}")
    print(f"* rank: {exp_config.rank}")

    model = MFConv(
        X_train=X, 
        V_init=get_basis(model_config.init_basis, rank=exp_config.rank, X=X), 
        R=model_config.R, 
        J=model_config.J, 
        K=model_config.K,
        lambda1=model_config.lambda1, 
        lambda2=model_config.lambda2, 
        lambda3=model_config.lambda3,
        rank=exp_config.rank
    )

    return model


def get_mf_lars_model(X, exp_config, model_config):

    print("Instantiating LARS MF model:")

    if model_config.max_iter > exp_config.rank:
        warnings.warn(f"--*-- ERROR: max_iter = {model_config.max_iter} > "
                       "rank = {exp_config.rank} is invalid configuration."
                       "Experiment will not be executed. --*--")

    print(f"* max_iter: {model_config.max_iter}")
    print(f"* lambda2: {model_config.lambda2}")
    print(f"* lambda3: {model_config.lambda3}")
    print(f"* rank: {exp_config.rank}")

    model = MFLars(
        X_train=X, 
        V_init=get_basis(model_config.init_basis, rank=exp_config.rank, X=X), 
        R=model_config.R, 
        J=model_config.J, 
        K=model_config.K,
        lambda2=model_config.lambda2, 
        lambda3=model_config.lambda3,
        max_iter=model_config.max_iter,
        rank=exp_config.rank
    )

    return model


def get_weighted_mf_tv_model(X, exp_config, model_config):

    print("Instantiating Weighted TV MF model:")
    print(f"* lambda1: {model_config.lambda1}")
    print(f"* lambda2: {model_config.lambda2}")
    print(f"* lambda3: {model_config.lambda3}")
    print(f"* gamma: {model_config.gamma}")
    print(f"* num_iter: {model_config.num_iter}")
    print(f"* rank: {exp_config.rank}")

    model = WeightedMFTV(
        X_train=X, 
        V_init=get_basis(model_config.init_basis, rank=exp_config.rank, X=X), 
        R=model_config.R, 
        W=get_weight_matrix(model_config.weighting, X=X),
        J=model_config.J, 
        gamma=model_config.gamma,
        lambda1=model_config.lambda1, 
        lambda2=model_config.lambda2, 
        lambda3=model_config.lambda3,
        num_iter=model_config.num_iter,
        rank=exp_config.rank
    )

    return model


def get_mf_tv_model(X, exp_config, model_config):

    print("Instantiating TV MF model:")
    print(f"* lambda1: {model_config.lambda1}")
    print(f"* lambda2: {model_config.lambda2}")
    print(f"* lambda3: {model_config.lambda3}")
    print(f"* gamma: {model_config.gamma}")
    print(f"* num_iter: {model_config.num_iter}")
    print(f"* rank: {exp_config.rank}")

    model = MFTV(
        X_train=X, 
        V_init=get_basis(model_config.init_basis, rank=exp_config.rank, X=X), 
        R=model_config.R, 
        J=model_config.J, 
        gamma=model_config.gamma,
        lambda1=model_config.lambda1, 
        lambda2=model_config.lambda2, 
        lambda3=model_config.lambda3,
        num_iter=model_config.num_iter,
        rank=exp_config.rank
    )

    return model

        
def init_model(X, exp_config, model_config, subset_idx=None):

    if model_config.model_type == "MFConv":
        return get_mf_conv_model(X=X, exp_config=exp_config, model_config=model_config)

    if model_config.model_type == "WMFConv":
        return get_weighted_mf_conv_model(X=X, exp_config=exp_config, model_config=model_config)

    if model_config.model_type == "MFLars":
        return get_mf_lars_model(X=X, exp_config=exp_config, model_config=model_config)

    if model_config.model_type == "MFTV":
        return get_mf_tv_model(X=X, exp_config=exp_config, model_config=model_config)

    if model_config.model_type == "WMFTV":
        return get_weighted_mf_tv_model(X=X, exp_config=exp_config, model_config=model_config)

    if model_config.model_type == "GDL":
        return get_gdl_model(X=X, exp_config=exp_config, model_config=model_config, subset_idx=subset_idx)

    raise ValueError(f"Unknown model: {model_config.name}")
