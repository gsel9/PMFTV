# Guidelines on when to use CV:
#   "Just wanted to add some simple guidelines that Andrew Ng mentioned ..."
#   REF: https://stats.stackexchange.com/questions/104713/hold-out-validation-vs-cross-validation
import time 

import numpy as np

from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

from .data.splitting import train_val_split

from .models.inference.map import MAP
from .models.model_generator import init_model

from .utils.metrics import model_performance


def train_mf_model(exp_config, run_config, model_config, X_train):

    model = init_model(X=X_train, exp_config=exp_config, model_config=model_config)

    print("-- Training model --")
    print(f'Running {exp_config.num_epochs} epochs.\n')

    start_time = time.time()
    for epoch in tqdm(range(exp_config.num_epochs)):

        model.train()

        loss_value = float(model.loss())
        #print(loss_value)
        if exp_config.monitor_loss:
            run_config.loss.append(loss_value)

        # NOTE: Assumes model improves on each iteration.
        run_config.U = model.U
        run_config.V = model.V
        run_config.M_hat = model.U @ model.V.T 

        if epoch % exp_config.epochs_per_display == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, loss_value))
            print()

    duration = time.time() - start_time
    run_config.update_config({"duration": duration})
    print(f"-- Training finished in {duration} --")

    return model


def train_mf_model_early_stopping(exp_config, run_config, model_config, X_train, val_set):

    print("-- Initialising model --")
    model = init_model(X=X_train, exp_config=exp_config, model_config=model_config)

    print("-- Training model with early stopping --")
    print(f'Running {exp_config.num_epochs} epochs.\n')

    max_score = -1
    patience_count = exp_config.patience

    start_time = time.time()
    for epoch in tqdm(range(exp_config.num_epochs)):

        model.train()

        M_hat = model.U @ model.V.T

        loss_value = float(model.loss())
        run_config.loss.append(loss_value)

        if epoch % exp_config.epochs_per_display == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, loss_value))
            print()

        if epoch % exp_config.epochs_per_val == 0:

            start_pred = time.time()
            score = predict(M_hat, X_val, val_set)
            run_config.append_value("mcc", score)

            if score > max_score:
                max_score = score
                run_config.M_hat = M_hat
            else:
                patience_count -= 1
                
                if patience_count == 0:
                    print("*** --- EARLY STOPPING --- ***")
                    print(f"Validation error increasing after {epoch + 1} epochs")

                    break 

    duration = time.time() - start_time
    run_config.update_config({"duration": duration})
    print(f"-- Training finished in {duration} --")

    return model


def predict(M_hat, X_val, val_set):

    estimator = MAP(M_train=M_hat, theta=2.5)
    y_pred = estimator.predict(val_set.X_train, val_set.time_of_prediction)

    return matthews_corrcoef(val_set.y_true, y_pred)
