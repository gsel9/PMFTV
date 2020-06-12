import numpy as np

from sklearn.metrics import (
    matthews_corrcoef, recall_score, accuracy_score
)


def rec_mse(M_true, O_train, M_pred):
        
    # Complement to O_train.
    O_train_c = np.ones_like(O_train) - O_train

    norm = np.size(M_true) * (1 - np.mean(O_train))
    
    return np.linalg.norm(O_train_c * (M_true - M_pred)) ** 2 / norm


def binary_confusion_matrix(X_true, O_val, X_pred, thresh):
    
    I, J = O_val.nonzero()
    
    cmat = np.zeros((2, 2), dtype=np.int32)
    for k in range(len(I)):
        
        true = int(X_true[I[k], J[k]] > thresh)
        pred = int(X_pred[I[k], J[k]] > thresh)
        
        cmat[true, pred] += 1
        
    return cmat


def multiclass_confusion_matrix(X_true, O_val, X_pred):
    
    I, J = O_val.nonzero()

    cmat = np.zeros((4, 4), dtype=np.int32)
    for k in range(len(I)):
        
        true = int(round(X_true[I[k], J[k]]))
        pred = int(round(X_pred[I[k], J[k]]))
        
        cmat[true - 1, pred - 1] += 1
        
    return cmat


def classification_report(x_true, x_pred, thresh=2):

    scores = {
        'mcc': matthews_corrcoef(x_true, x_pred),
        'mcc_binary': matthews_corrcoef(x_true > thresh, x_pred > thresh),
        'recall_micro': recall_score(x_true, x_pred, average="micro", zero_division=0),
        #'roc_auc_score_micro': roc_auc_score(x_true, x_pred, average="micro")
        'accuracy': accuracy_score(x_true > thresh, x_pred > thresh),
        'sensitivity': recall_score(x_true > thresh, x_pred > thresh, average="binary", zero_division=0),
        'specificity': recall_score(x_true < thresh, x_pred < thresh, average="binary", zero_division=0),
    }

    return scores


def model_performance(y_true, y_pred, run_config=None):

    scores = classification_report(y_true, y_pred)

    run_config.append_values(scores)

    return scores
