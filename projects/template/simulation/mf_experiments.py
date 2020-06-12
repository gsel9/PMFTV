# Guidelines on when to use CV:
#   "Just wanted to add some simple guidelines that Andrew Ng mentioned ..."
#   REF: https://stats.stackexchange.com/questions/104713/hold-out-validation-vs-cross-validation
import time 

import numpy as np

from sklearn.metrics import matthews_corrcoef

from .configs import RunConfig

from .mf_training import train_mf_model, train_mf_model_early_stopping

from .data.dataset import KFoldDataset
from .data.load_data import load_data_from_file
from .data.splitting import train_val_split

from .utils.metrics import model_performance
from .utils.save_results import save_to_disk

from .models.inference.map import MAP


def validation_performance(run_config, X_train, y_true, time_of_prediction):

    print("-- Predicting --")
    estimator = MAP(M_train=run_config.M_hat, theta=2.5)
    y_pred = estimator.predict(X_train, time_of_prediction)

    print("-- Performance scores --")
    scores = model_performance(y_true, y_pred, run_config)
    print("MCC: {}; MCC binary: {}; Sensitivity: {}\n".format(
        scores["mcc"], scores["mcc_binary"], scores["sensitivity"])
    )    


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
        
        _ = train_mf_model(exp_config, run_config, model_config, X_train)

        validation_performance(run_config, kfolds.X_test, kfolds.y_true, 
                               kfolds.time_of_prediction)

    save_to_disk(exp_config=exp_config, run_config=run_config, model_config=model_config)


def matrix_completion(exp_config, model_config):

    print(f"Initiated experiment: `{exp_config.exp_id}`")
    print("-" * 30, "\n")

    run_config = RunConfig()

    _, X_train = load_data_from_file(exp_config, run_config=run_config)

    val_set = None
    if exp_config.val_size > 0:
        X_train, val_set = train_val_split(X_train, exp_config) 

    if exp_config.early_stopping:

        if val_set is None:
            raise ValueError("Should specify size validation set for early stopping.")

        model = train_mf_model_early_stopping(exp_config=exp_config, 
                                              run_config=run_config, 
                                              model_config=model_config, 
                                              X=X_train, val_set=val_set)
    else:     
        model = train_mf_model(exp_config, run_config, model_config, X_train)

    if hasattr(model_config, "alphas"):
        model_config.update_value("alphas", model.alphas)

    if val_set is not None:
        validation_performance(run_config, val_set.X_train, val_set.y_true, 
                               val_set.time_of_prediction)

    save_to_disk(exp_config=exp_config, run_config=run_config, model_config=model_config)
