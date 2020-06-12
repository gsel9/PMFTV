import warnings

import numpy as np

from .gdl import MGCNN
from .matrix_factorisation import MFConv, MFLars, MFTV


def get_basis(init_basis, rank, X):

    if init_basis == "svd":

        _, s, V = np.linalg.svd(X, full_matrices=False)
        return np.dot(U[:, :rank], np.diag(np.sqrt(s[:rank])))

    if init_basis == "random":
        np.random.seed(42)
        return np.random.choice(range(1, 5), size=(X.shape[1], rank), p=(0.9, 0.05, 0.04, 0.01))

    if init_basis == "hmm":

        np.random.seed(42)
        data = np.load("/Users/sela/Desktop/tsd_code/data/hmm/base_set_300K.npy")
        idx = np.random.choice(range(data.shape[0]), size=rank, replace=False)

        return np.transpose(data[idx])

    if init_basis == "smooth_hmm":

        np.random.seed(42)
        data = np.load("/Users/sela/Desktop/tsd_code/data/hmm/base_set_300K.npy")
        idx = np.random.choice(range(data.shape[0]), size=rank, replace=False)

        return np.transpose(data[idx])


def get_gdl_model(X, exp_config, model_config, L_row, L_col):

    print("Instantiating GDL model:")
    print(f"* n_conv_feat: {model_config.n_conv_feat}")
    print(f"* ord_row_conv: {model_config.ord_row_conv}")
    print(f"* ord_col_conv: {model_config.ord_col_conv}")
    print(f"* diffusion_steps: {model_config.diffusion_steps}")
    print(f"* rank: {exp_config.rank}")

    model =  MGCNN(
        X=X,
        Lr=L_row, 
        Lc=L_col,
        rank=exp_config.rank,
        domain=exp_config.domain,
        n_conv_feat=model_config.channels, 
        ord_row_conv=model_config.degree_row_poly, 
        ord_col_conv=model_config.degree_col_poly,
        diffusion_steps=model_config.diffusion_steps
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

        
def init_model(X, exp_config, model_config, L_row=None, L_col=None):

    if model_config.model_type == "MFConv":
        return get_mf_conv_model(X=X, exp_config=exp_config, model_config=model_config)

    if model_config.model_type == "MFLars":
        return get_mf_lars_model(X=X, exp_config=exp_config, model_config=model_config)

    if model_config.model_type == "MFTV":
        return get_mf_tv_model(X=X, exp_config=exp_config, model_config=model_config)

    if model_config.model_type == "GDL":
        return get_gel_model(X=X, exp_config=exp_config, model_config=model_config, 
                             L_row=L_row, L_col=L_col)

    raise ValueError(f"Unknown model: {model_config.name}")
