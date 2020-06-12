import os
import time

import numpy as np
import tensorflow as tf

from .save_to_disk import save_results
from .models.model_generator import init_model
from .metrics import multiclass_confusion_matrix, frobenius_norm, classification_report
from .configs import RunConfig
from .data.train_val_split import train_val_split_from_file
from .graphs.laplacian_generator import row_col_laplacian
from .loss.gdl_loss import set_loss


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def run_gdl_kfold_experiment(model_config, data_config, row_graph_config, 
                             col_graph_config, exp_config, train_config):

    print(f"Initiated experiment: `{exp_config.exp_id}`")
    print("-" * 30, "\n")
    print("Model config:")
    print(f"* Row gamma: {model_config.row_gamma}")
    print(f"* Col gamma: {model_config.col_gamma}")
    print(f"* Row poly degree: {model_config.degree_row_poly}")
    print(f"* Col poly degree: {model_config.degree_col_poly}")
    print(f"* Channels: {model_config.channels}")
    print(f"* Diffusion steps: {model_config.diffusion_steps}\n")

    X = np.load(f"{data_config.path_to_datadir}/X_train.npy")
    print(f"* \nLoaded {np.shape(X)} data matrix from:")
    print(data_config.path_to_datadir)

    if data_config.num_samples is not None:
        print(f"* Selecting subset of {data_config.num_samples} samples")
        
        np.random.seed(SEED)
        idx = np.random.choice(range(X.shape[0]), replace=False, size=data_config.num_samples)
        X = X[idx]

        v, c = np.unique(X[X != 0], return_counts=True)
        print("* Subset:\n- Values {}; Counts {}".format(v, c))

    kfolds = TemporalDatasetKFold(X, n_splits=train_config.cv_splits, 
                                  time_lag=train_config.time_lag)

    print(f"\nInitiating K-fold CV:")
    for i in range(train_config.cv_splits):

        kfolds.i_fold = i
        print(f"* fold = {kfolds.i_fold}\n")

        # Lacplacians from a subset of the row graph adjacency matrix.         
        L_row, L_col = row_col_laplacian(row_graph_config, col_graph_config, row_subset_idx=kfolds.train_rows_idx)
        print(f"Computed {L_row.shape} row graph Laplacian and {L_col.shape} col graph Laplacian.\n")

        print("Initialising model")
        model = init_model(X_train=kfolds.X_train, data_config=data_config, model_config=model_config,
                           L_row=L_row, L_col=L_col)

        print("Training model:")
        print("-" * 30, "\n")
        train_gdl_model(train_config=train_config, model_config=model_config, model=model)

        print("Predicting scores:")
        estimator = Predict(M_train=train_config.opt_X, theta=2.5)
        y_pred = estimator.predict(kfolds.X_test, kfolds.time_of_prediction)

        scores = model_performance(kfolds.y_true, y_pred, train_config)
        print("Performance scores:")
        print("MCC: {}; Sensitivity: {}\n".format(scores["mcc"], scores["sensitivity"]))

    save_results(model_config=model_config,
                 data_config=data_config,
                 exp_config=exp_config,
                 train_config=train_config,
                 row_graph_config=row_graph_config,
                 col_graph_config=col_graph_config)


###################


def kfold_model_selection(exp_config, model_config, row_graph_config, col_graph_config):

    print(f"Experiment: `{exp_config.exp_id}`")

    run_config = RunConfig()

    X = np.load(f"{data_config.path_to_datadir}/train/X_train.npy")
    print(f"* \nLoaded {np.shape(X)} data matrix from:")
    print(data_config.path_to_datadir)

    train_idx = None
    if exp_config.num_train_samples is not None:
        print(f"Selecting subset of {exp_config.num_train_samples} training samples")

        np.random.seed(exp_config.seed)
        train_idx = np.random.choice(range(X.shape[0]), replace=False, size=exp_config.num_train_samples)
        X = X[train_idx]

        v, c = np.unique(X[X != 0], return_counts=True)
        print("Values {}; Counts {}".format(v, c))

        run_config.update_config({"unique_train_values": v, "train_distribution": c})

    kfolds = KFoldDataset(X, n_splits=exp_config.cv_splits, prediction_window=exp_config.prediction_window)

    print(f"Running {train_config.cv_splits} folds")
    for i in range(train_config.cv_splits):

        # Lacplacians from a subset of the row graph adjacency matrix.         
        L_row, L_col = row_col_laplacian(row_graph_config, col_graph_config, row_subset_idx=kfolds.train_idx)
        print(f"Computed {L_row.shape} row graph Laplacian and {L_col.shape} col graph Laplacian.\n")

        print("Initialising model")
        model = init_model(X_train=kfolds.X_reconstruct, data_config=data_config, model_config=model_config,
                           L_row=L_row, L_col=L_col, W_init=W_init, H_init=H_init)
    
        print("Reconstructing data matrix")
        run_gdl(train_set=kfold.rec_obj, val_set=kfold.pred_obj, exp_config=exp_config, run_config=run_config, 
                model_config=model_config, Lr=Lr, Lc=Lc)

    save_results(model_config=model_config,
                 run_config=run_config,
                 exp_config=exp_config,
                 row_graph_config=row_graph_config,
                 col_graph_config=col_graph_config)


def run_gdl(train_set, val_set, exp_config, run_config, model_config, Lr, Lc):

    loss = set_loss(model_config=model_config, X_true=train_set.X_reconstruct, O=train_set.O_reconstruct, Lr=Lr, Lc=Lc)
    model = init_model(X=train_set.X_reconstruct, exp_config=exp_config, model_config=model_config, L_row=Lr, L_col=Lc)
    optimizer = init_optimiser(model_config)

    opt_val_mcc = -1

    print(f'Running {exp_config.num_epochs} epochs.')

    start_time = time.time()
    for epoch in range(exp_config.num_epochs):

        loss_value, grads, M_hat_adj = train_step(model, optimizer, loss)
        
        if epoch % exp_config.epochs_per_display == 0:

            norm_grad = np.linalg.norm(tf.concat([tf.reshape(grad, [-1]) for grad in grads], 0))
            
            print('Epoch: {}, Loss: {}, Gradient: {}'.format(epoch, loss_value, norm_grad))

        if model.n_iter_ % exp_config.epochs_per_val == 0:

            X_true_sub, O_true_sub, X_pred_sub = model.predict(val_set, M_hat_adj, prediction_window=exp_config.prediction_window)

            scores = classification_report(X_true_sub, O_true_sub, X_pred_sub, thresh=2)

            print(multiclass_confusion_matrix(X_true_sub, O_true_sub, X_pred_sub))

            if scores["mcc"] > opt_val_mcc:
                opt_val_mcc = scores["mcc"]

                run_config.update_config({"opt_epoch": epoch + 1})
                run_config.update_config(scores)

                run_config.model_weights = model.get_config()
                run_config.M_hat = M_hat_adj

            else:
                print("*** --- EARLY STOPPING --- ***")
                print(f"Validation error increasing after {epoch} epochs")

                break

        run_config.append_value("loss", loss_value)

    duration = time.time() - start_time
    run_config.append_value("duration", duration)
    print(f"Training finished in {duration}.")


def train_step(model, optimizer, loss):

    with tf.GradientTape() as tape:

        # Matrix completion.
        U, V = model(model.U_init, model.V_init)

        M_hat = tf.matmul(U, V, transpose_b=True)
        M_hat_adj = adjust_range(M_hat)

        loss_value = loss(M_pred=M_hat, M_pred_adj=M_hat_adj)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return float(loss_value), grads, M_hat_adj.numpy()


def adjust_range(Z, shift=1, scale=3):

    min_max_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    return shift + scale * min_max_norm


def init_optimiser(model_config):

    if model_config.optimiser == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=model_config.learning_rate)

    raise ValueError(f"Did not recognise optimiser: {model_config.optimiser}")
