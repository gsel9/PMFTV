import warnings

import numpy as np

from sklearn.utils import class_weight

from .gdl import MGCNN
from .matrix_factorisation import (
    MFConv, MFLars, MFTV, WeightedMFConv, WeightedMFTV
)


def get_basis(init_basis, rank, X):

    # NOTE: The initial approach. 
    if init_basis == "mean":

        return np.ones((X.shape[1], rank)) * np.mean(X[X.nonzero()])

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

    if init_basis == "smooth-hmm":

        np.random.seed(42)
        data = np.load("/Users/sela/Desktop/tsd_code/data/hmm/base_set_300K.npy")
        idx = np.random.choice(range(data.shape[0]), size=rank, replace=False)

        return np.transpose(data[idx])

    if init_basis == "noisy-logarithm":

        np.random.seed(42)
        # Half of profiles are logarithm functions and half are cancelling functions.
        # Add (standard) Gaussian noise for variability in the profiles.


def create_class_weight(X, mu=0.15):

    N = np.count_nonzero(X)
    classes = np.unique(X[X != 0])

    class_weight = []
    for c in classes:
        score = np.log(mu * N / np.sum(X == c))
        weight = score if score > 1.0 else 1.0
        class_weight.append(weight)

    return class_weight / sum(class_weight)


def get_weight_matrix(weighting, X):

    if weighting == "identity":
        return np.eye(X.shape[0])

    if weighting == "max-state":
        return np.diag(np.max(X, axis=1))

    # Current optimal.
    if weighting == "max-state-scaled":
        return np.diag(np.max(X, axis=1)) / 2

    if weighting == "max-state-scaled-max":
        return np.diag(np.max(X, axis=1)) / 4

    if weighting == "max-state-normed":
        W = np.diag(np.max(X, axis=1))
        return W / np.linalg.norm(W)

    if weighting == "binary":
        W_id = np.diag(np.max(X, axis=1))
        W = np.eye(X.shape[0])
        W[W_id > 2] = 2
        return W

    if weighting == "scaled-norm":
        W = np.linalg.norm(X, axis=1)
        return np.diag(W / max(W))

    if weighting == "sklearn-balanced":
        weights = class_weight.compute_class_weight('balanced', np.unique(X[X != 0]), X[X != 0])
        weights = weights / sum(weights)

        W = np.diag(np.max(X, axis=1))
        W[W == 1] = weights[0]
        W[W == 2] = weights[1]
        W[W == 3] = weights[2]
        W[W == 4] = weights[3]
        return W

    if weighting == "custom-balanced":
        weights = create_class_weight(X)

        W = np.diag(np.max(X, axis=1))
        W[W == 1] = weights[0]
        W[W == 2] = weights[1]
        W[W == 3] = weights[2]
        W[W == 4] = weights[3]
        return W

    if weighting == "normalised":
        return np.diag(np.max(X, axis=1)) / 4

    # Failed.
    if weighting == "log-max-state-scaled-max":
        w = 1 + np.log(np.max(X, axis=1))
        return np.diag(w / max(w))

    if weighting == "relative":
        m = np.max(X, axis=1)
        vals, counts = np.unique(m, return_counts=True)
        weights = {int(v): counts[np.argmax(counts)] / counts[i] for i, v in enumerate(vals)}
        w = np.zeros(X.shape[0])
        for c, weight in weights.items():
            w[m == int(c)] = weight

        return np.diag(w)


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

        
def init_model(X, exp_config, model_config, L_row=None, L_col=None):

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
        return get_gel_model(X=X, exp_config=exp_config, model_config=model_config, 
                             L_row=L_row, L_col=L_col)

    raise ValueError(f"Unknown model: {model_config.name}")
