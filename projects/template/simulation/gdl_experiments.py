import os
import time

import numpy as np
import tensorflow as tf

from .utils.save_to_disk import save_results
from .utils.metrics import multiclass_confusion_matrix, classification_report
from .models.model_generator import init_model
from .configs import RunConfig
from .data.train_val_split import train_val_split_from_file
from .models.loss.gdl_loss import set_loss
from .models.gdl.graphs.laplacian_generator import row_col_laplacian


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def matrix_completion(exp_config, model_config, row_graph_config, col_graph_config):

    print(f"Experiment: `{exp_config.exp_id}`")

    run_config = RunConfig()

    train_set, val_set, idx = train_val_split_from_file(exp_config=exp_config, run_config=run_config, return_train_idx=True)

    Lr, Lc = row_col_laplacian(row_graph_config, col_graph_config, row_subset_idx=idx)
    print(f"Computed {Lr.shape} row graph Laplacian and {Lc.shape} col graph Laplacian.\n")

    print("Reconstructing data matrix")
    run_gdl(train_set=train_set, val_set=val_set, exp_config=exp_config, run_config=run_config, 
            model_config=model_config, Lr=Lr, Lc=Lc)

    save_results(model_config=model_config,
                 run_config=run_config,
                 exp_config=exp_config,
                 row_graph_config=row_graph_config,
                 col_graph_config=col_graph_config)


def run_gdl(train_set, val_set, exp_config, run_config, model_config, Lr, Lc):

    loss_fn = set_loss(model_config=model_config, X_true=train_set.X, O=train_set.O_train, Lr=Lr, Lc=Lc)
    model = init_model(X=train_set.X, exp_config=exp_config, model_config=model_config, L_row=Lr, L_col=Lc)
    optimizer = init_optimiser(model_config)

    model.compile(optimizer=optimizer, loss=loss_fn)

    opt_val_mcc = -1

    print(f'Running {exp_config.num_epochs} epochs.')

    start_time = time.time()
    for epoch in range(exp_config.num_epochs):

        loss_value, grads, M_hat_adj = train_step(model, optimizer, loss_fn)

        M_hat_adj = M_hat_adj.numpy()
        loss_value = float(loss_value)
        
        if epoch % exp_config.epochs_per_display == 0:

            norm_grad = np.linalg.norm(tf.concat([tf.reshape(grad, [-1]) for grad in grads], 0))
            
            print('Epoch: {}, Loss: {}, Gradient: {}'.format(epoch, loss_value, norm_grad))

        if epoch % exp_config.epochs_per_val == 0:

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
    
    # Record validation set training and test masks.
    run_config.X_true = X_true_sub
    run_config.X_pred = X_pred_sub
    run_config.O_val = O_true_sub

    duration = time.time() - start_time
    run_config.append_value("duration", duration)
    print(f"Training finished in {duration}.")


@tf.function
def train_step(model, optimizer, loss_fn):

    #np.random.seed(42)
    #U_init = np.float32(np.random.normal(0.0, np.sqrt(2.0 / np.size(model.X)), size=(model.X.shape[0], model.rank)))
    #V_init = np.float32(np.random.normal(0.0, np.sqrt(2.0 / np.size(model.X)), size=(model.X.shape[1], model.rank)))
    #U_init = (U_init - np.mean(U_init)) / np.std(U_init)
    #V_init = (V_init - np.mean(V_init)) / np.std(V_init)

    U, s, V = np.linalg.svd(model.X, full_matrices=0)
    rank_W_H = 5
    partial_s = s[:rank_W_H]
    partial_S_sqrt = np.diag(np.sqrt(partial_s))
    U_init = np.float32(np.dot(U[:, :rank_W_H], partial_S_sqrt))
    V_init = np.float32(np.dot(partial_S_sqrt, V[:rank_W_H, :]).T)

    #from sklearn.decomposition import NMF
    #nmf = NMF(n_components=5, init="random", tol=0.0001, max_iter=200, random_state=42, shuffle=False)
    #U_init = nmf.fit_transform(model.X)
    #V_init = np.transpose(nmf.components_)

    with tf.GradientTape() as tape:

        tape.watch(model.trainable_weights)

        # Matrix completion.
        U, V = model([U_init, V_init])

        M_hat = tf.matmul(U, V, transpose_b=True)
        M_hat_adj = adjust_range(M_hat)

        loss_value = loss_fn(M_pred=M_hat, M_pred_adj=M_hat_adj)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value, grads, M_hat_adj


@tf.function
def adjust_range(Z, shift=1, scale=3):

    min_max_norm = (Z - tf.reduce_min(Z)) / (tf.reduce_max(Z) - tf.reduce_min(Z))

    return shift + scale * min_max_norm


def init_optimiser(model_config):

    if model_config.optimiser == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=model_config.learning_rate)

    raise ValueError(f"Did not recognise optimiser: {model_config.optimiser}")
