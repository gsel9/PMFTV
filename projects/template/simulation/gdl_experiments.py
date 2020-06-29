import os
import time 

import numpy as np
import tensorflow as tf

from .configs import RunConfig

from .gdl_training import (
    train_gdl_model, 
    train_gdl_model_early_stopping
)

from .data.dataset import KFoldDataset
from .data.load_data import load_data_from_file
from .data.splitting import train_val_split

from .utils.save_results import save_to_disk

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# NOTE: This function can be used with both MF And GDL.
def kfold_matrix_completion(exp_config, model_config):
    # NB: KFoldDataset and TrainTestDataset should have same API.

    print(f"Initiated experiment: `{exp_config.exp_id}`")
    print("-" * 30, "\n")

    run_config = RunConfig()

    idx, X = load_data_from_file(exp_config, run_config=run_config)

    print(f"Initiating {exp_config.n_kfold_splits}-fold CV:")
    kfolds = KFoldDataset(X, n_splits=exp_config.n_kfold_splits, time_lag=exp_config.time_lag)
    for i in range(exp_config.n_kfold_splits):

        kfolds.i_fold = i
        print(f"Fold = {kfolds.i_fold}")

        if exp_config.early_stopping:

            if exp_config.patience > exp_config.num_epochs:
                raise ValueError("`patience` > num_epochs")

            model = train_gdl_model_early_stopping(exp_config=exp_config, 
                                                   run_config=run_config, 
                                                   model_config=model_config, 
                                                   X_train=kfolds.X_rec, val_set=kfolds)
        else:
            _ = train_gdl_model(exp_config, run_config, model_config, 
                                kfolds.X_rec, val_set=kfolds)

    save_to_disk(exp_config=exp_config, run_config=run_config, model_config=model_config)


def matrix_completion(exp_config, model_config):

    print(f"Initiated experiment: `{exp_config.exp_id}`")
    print("-" * 30, "\n")

    run_config = RunConfig()

    # TODO: check if exp_config.num_val_samples > 0 and return val_set (defaults to None).
    idx, X_rec = load_data_from_file(exp_config, run_config=run_config)

    val_set = None
    if exp_config.val_size > 0:
        X_rec, val_set = train_val_split(X_rec, exp_config) 

    if exp_config.early_stopping:

        if val_set is None:
            raise ValueError("Should specify size validation set for early stopping.")

        if exp_config.patience > exp_config.num_epochs:
            raise ValueError("`patience` > num_epochs")

        model = train_gdl_model_early_stopping(exp_config=exp_config, 
                                               run_config=run_config, 
                                               model_config=model_config, 
                                               subset_idx=idx, X_train=X_rec, val_set=val_set)
    else:
        model = train_gdl_model(exp_config, run_config, model_config, X_rec,
                                subset_idx=idx, val_set=val_set)

    # NOTE: Lasso-Lars model.
    if hasattr(model_config, "alphas"):
        model_config.update_value("alphas", model.alphas)

    save_to_disk(exp_config=exp_config, run_config=run_config, model_config=model_config)
