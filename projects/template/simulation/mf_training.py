# Guidelines on when to use CV:
#   "Just wanted to add some simple guidelines that Andrew Ng mentioned ..."
#   REF: https://stats.stackexchange.com/questions/104713/hold-out-validation-vs-cross-validation
import time 

import numpy as np

from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

from .models.inference.map import MAP
from .models.model_generator import init_model


def train_mf_model(exp_config, run_config, model_config, X_train, val_set=None):

    model = init_model(X=X_train, exp_config=exp_config, model_config=model_config)

    print("-- Training model --")
    print(f'Running {exp_config.num_epochs} epochs.\n')

    start_time = time.time()
    for epoch in tqdm(range(exp_config.num_epochs)):

        model.train()

        loss_value = float(model.loss())
        if exp_config.monitor_loss:
            run_config.loss.append(loss_value)

        run_config.U = model.U
        run_config.V = model.V
        run_config.M_hat = model.U @ model.V.T 

        if epoch % exp_config.epochs_per_display == 0:
            print('Epoch: {}, Loss: {}\n'.format(epoch, loss_value))

    duration = time.time() - start_time
    run_config.update_config({"duration": duration})
    print(f"-- Training finished in {duration} --")

    if val_set is not None:
        run_config.append_value("mcc", validation_performance(run_config, val_set, optimise="mcc"))

    return model


def train_mf_model_early_stopping(exp_config, run_config, model_config, X_train, val_set=None):

    print("-- Initialising model --")
    model = init_model(X=X_train, exp_config=exp_config, model_config=model_config)

    print("-- Training model with early stopping --")
    print(f'Running {exp_config.num_epochs} epochs.\n')

    prev_score = -1
    prev_loss = np.inf

    chances_cnt = exp_config.chances_to_improve

    start_time = time.time()
    for epoch in tqdm(range(exp_config.num_epochs)):

        model.train()

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
