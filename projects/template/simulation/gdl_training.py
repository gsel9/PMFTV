import os
import time 

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

from .models.gdl import get_gdl_loss_fn, get_gdl_optimiser
from .models.inference.map import MAP
from .models.model_generator import init_model

from .utils.special_matrices import get_coefs, get_basis

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@tf.function
def adjust_range(Z, shift=1, scale=3):

    min_max_norm = (Z - tf.reduce_min(Z)) / (tf.reduce_max(Z) - tf.reduce_min(Z))

    return shift + scale * min_max_norm


@tf.function
def train_step(model, optimiser, loss_fn, X_train, V_init, U_init):

    with tf.GradientTape() as tape:

        tape.watch(model.trainable_weights)

        # Matrix completion.
        U, V = model([U_init, V_init])

        M_hat = tf.matmul(U, V, transpose_b=True)
        M_hat_adj = adjust_range(M_hat)

        loss_value = loss_fn(Y_true=X_train, M_pred=M_hat, M_pred_adj=M_hat_adj)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimiser.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value, grads, M_hat_adj


# TODO: Plot grads.
def train_gdl_model(exp_config, run_config, model_config, X_train, subset_idx=None, val_set=None):

    model = init_model(X=X_train, exp_config=exp_config, model_config=model_config, subset_idx=subset_idx)
    
    loss_fn = get_gdl_loss_fn(model_config.loss, model.Lr, model.Lc, 
                              model_config.col_gamma, model_config.row_gamma)

    optimiser = get_gdl_optimiser(model_config.optimiser, model_config.learning_rate)

    V_init = get_basis(model_config.init_basis, rank=exp_config.rank, X=X_train) 
    U_init = get_coefs(model_config.init_coefs, rank=exp_config.rank, X=X_train)

    print("-- Training model --")
    print(f'Running {exp_config.num_epochs} epochs.\n')

    start_time = time.time()
    for epoch in tqdm(range(exp_config.num_epochs)):

        loss_value, grads, M_hat = train_step(model, optimiser, loss_fn, X_train, V_init, U_init)

        if exp_config.monitor_loss:
            run_config.loss.append(loss_value)

        run_config.U = model.U
        run_config.V = model.V
        run_config.M_hat = M_hat

        if epoch % exp_config.epochs_per_display == 0:
            print('Epoch: {}, Loss: {}\n'.format(epoch, loss_value))

    duration = time.time() - start_time
    run_config.update_config({"duration": duration})
    print(f"-- Training finished in {duration} --")

    if val_set is not None:
        run_config.append_value("mcc", validation_performance(run_config, val_set, optimise="mcc"))

    return model


def train_gdl_model_early_stopping(exp_config, run_config, model_config, X_train, subset_idx=None, val_set=None):

    print("-- Initialising model --")
    model = init_model(X=X_train, exp_config=exp_config, model_config=model_config, subset_idx=subset_idx)

    loss_fn = get_gdl_loss_fn(self.config["loss"], self.L_row, self.L_col, self.col_gamma, self.row_gamma)

    optimiser = get_gdl_optimiser(self.config["optimiser"], self.config["learning_rate"])

    V_init = get_basis(model_config.init_basis, rank=exp_config.rank, X=X_train) 
    U_init = get_coefs(model_config.init_coefs, rank=exp_config.rank, X=X_train)

    print("-- Training model with early stopping --")
    print(f'Running {exp_config.num_epochs} epochs.\n')

    prev_score = -1
    prev_loss = np.inf

    chances_cnt = exp_config.chances_to_improve

    start_time = time.time()
    for epoch in tqdm(range(exp_config.num_epochs)):

        loss_value, grads, M_hat = train_step(model, model_config, V_init, U_init)

        if epoch == exp_config.patience:
            run_config.U = model.U
            run_config.V = model.V
            run_config.M_hat = model.U @ model.V.T

        loss_value = float(model.loss())
        if exp_config.monitor_loss:
            run_config.loss.append(loss_value)

        if epoch % exp_config.epochs_per_display == 0:
            print('Epoch: {}, Loss: {}\n'.format(epoch, loss_value))

        if epoch % exp_config.epochs_per_val == 0 and epoch > exp_config.patience:

            score_value = validation_performance(run_config, val_set, optimise="mcc")
            if score_value < prev_score:
                if chances_cnt <= 0:
                    print(f"!!! EARLY STOPPING !!!: Score decreasing from {prev_score} to {score_value}")
                    break
                else:
                    print(f"!!! Score decreasing from {prev_score} to {score_value} !!!:\nChances left: {chances_cnt}")
                    chances_cnt -= 1

            if prev_loss < loss_value:
                if chances_cnt <= 0:
                    print(f"!!! EARLY STOPPING !!!: Loss increasing from {prev_loss} to {loss_value}")
                    break
                else:
                    print(f"!!! Loss increasing from {prev_loss} to {loss_value} !!!:\nChances left: {chances_cnt}")
                    chances_cnt -= 1

            run_config.U = model.U
            run_config.V = model.V
            run_config.M_hat = model.U @ model.V.T
            run_config.append_value("mcc", score_value)

            if abs(prev_loss - loss_value) < exp_config.tol:
                print("!!! EARLY STOPPING !!!: Converged with loss update"
                      f" {abs(prev_loss - loss_value)}")
                break

            prev_loss = loss_value 
            prev_score = score_value
            
    exp_config.update_value("num_epochs", epoch + 1)

    duration = time.time() - start_time
    run_config.update_config({"duration": duration})
    print(f"-- Training finished in {duration} --")

    return model


# NOTE: This function can be shared by all training schemes.
def validation_performance(run_config, val_set, optimise="mcc"):

    print("-- Predicting --")
    estimator = MAP(M_train=run_config.M_hat, theta=2.5)
    y_pred = estimator.predict(val_set.X_train, val_set.time_of_prediction)

    if optimise == "mcc":
        return matthews_corrcoef(val_set.y_true, y_pred)
