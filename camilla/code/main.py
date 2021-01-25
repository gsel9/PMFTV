import time 
import numpy as np

from sklearn.metrics import matthews_corrcoef

from src.utils.special_matrices import (
    finite_difference_matrix, 
    laplacian_kernel_matrix
)
from src.data import TemporalDatasetTrain, TemporalDatasetPredict
from src.model import CMF
from src.metrics import mcc_score


def matrix_completion():

    T = 321

    R = finite_difference_matrix(T)
    K = laplacian_kernel_matrix(T)

    return CMF(lambda1=1.0, lambda2=1.0, lambda3=1.0, 
               R=5, theta=2.5, max_iter=100, tol=1e-6,
               T=T, D=R, K=K)


def main():

    X_train = np.load('../../data/screening/X_train.npy')
    X_test = np.load('../../data/screening/X_test.npy')

    print(f"Loaded {X_train.shape} training matrix")
    print(f"Loaded {X_test.shape} test matrix")

    # TEMP: Tp speed up demonstration.
    X_test = X_test[:10]
    ### 

    model = matrix_completion()

    t0 = time.time()
    model.fit(X_train)

    print(f"Matrix completion finished in {time.time() - t0}")

    test_data = TemporalDatasetPredict(X_test, prediction_rule='last_observed')

    t0 = time.time()
    y_pred = model.predict(test_data.X, test_data.time_of_prediction)
    print(f"Predictions finished in {time.time() - t0}")

    print("Score:", matthews_corrcoef(test_data.y_true, y_pred))

    # Save some results.
    np.save("../results/screening_data/M_hat.npy", model.U_ @ model.V_.T)
    np.save("../results/screening_data/y_pred.npy", y_pred)


if __name__ == '__main__':
    main()
