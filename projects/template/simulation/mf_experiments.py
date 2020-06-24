# Guidelines on when to use CV:
#   "Just wanted to add some simple guidelines that Andrew Ng mentioned ..."
#   REF: https://stats.stackexchange.com/questions/104713/hold-out-validation-vs-cross-validation
import time 

import numpy as np

from .configs import RunConfig

from .mf_training import (
    train_mf_model, 
    train_mf_model_early_stopping
)

from .data.dataset import KFoldDataset
from .data.load_data import load_data_from_file
from .data.splitting import train_val_split

from .utils.save_results import save_to_disk


def kfold_matrix_completion(exp_config, model_config):

    print(f"Initiated experiment: `{exp_config.exp_id}`")
    print("-" * 30, "\n")

    run_config = RunConfig()

    _, X = load_data_from_file(exp_config, run_config=run_config)

    print(f"Initiating K-fold CV:")
    kfolds = KFoldDataset(X, n_splits=exp_config.n_kfold_splits, time_lag=exp_config.time_lag)
    for i in range(exp_config.n_kfold_splits):

        kfolds.i_fold = i
        print(f"Fold = {kfolds.i_fold}")
        
        # NB: KFoldDataset and TrainTestDataset should have same API.
        _ = train_mf_model(exp_config, run_config, model_config, 
                           kfolds.X_rec, val_set=kfolds)

    save_to_disk(exp_config=exp_config, run_config=run_config, model_config=model_config)


def matrix_completion(exp_config, model_config):

    print(f"Initiated experiment: `{exp_config.exp_id}`")
    print("-" * 30, "\n")

    run_config = RunConfig()

    # TODO: check if exp_config.num_val_samples > 0 and return val_set (defaults to None).
    _, X_rec = load_data_from_file(exp_config, run_config=run_config)

    val_set = None
    if exp_config.val_size > 0:
        X_rec, val_set = train_val_split(X_rec, exp_config) 

    if exp_config.early_stopping:

        if val_set is None:
            raise ValueError("Should specify size validation set for early stopping.")

        if exp_config.patience > exp_config.num_epochs:
            raise ValueError("`patience` > num_epochs")

        model = train_mf_model_early_stopping(exp_config=exp_config, 
                                              run_config=run_config, 
                                              model_config=model_config, 
                                              X_train=X_rec, val_set=val_set)
    else:
        model = train_mf_model(exp_config, run_config, model_config, X_rec,
                               val_set=val_set)

    if hasattr(model_config, "alphas"):
        model_config.update_value("alphas", model.alphas)

    save_to_disk(exp_config=exp_config, run_config=run_config, model_config=model_config)
