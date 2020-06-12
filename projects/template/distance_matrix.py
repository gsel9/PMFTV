import time 
import numpy as np 

from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances

from utils.load_data import load_train_val
from utils.run_configs import DataConfig


def wasserstein_p1(x, y):
    
    sx, hx = np.unique(x[x != 0], return_counts=True)
    sy, hy = np.unique(y[y != 0], return_counts=True)

    return wasserstein_distance(sx, sy, hx / sum(hx), hy / sum(hy))


def approx_distance_matrix():
    # Sample N > K nodes.
    # Sample next node by K-means++ formula
    # Keep K closest.

    from tqdm import tqdm

    X_train = np.load("/Users/sela/Desktop/recsys_paper/data/screening/train/X_train.npy")

    # TEMP:
    #X_train = X_train[:5]

    print(f"-> Loaded {np.shape(X_train)} data matrix.")
    print("-> Computing distance matrix.")

    start_time = time.time()
    D = np.zeros((X_train.shape[0], X_train.shape[0]))
    for i, x in enumerate(tqdm(X_train)):

        np.random.seed(42)
        candidates = np.random.choice(range(X_train.shape[0]), replace=False, size=2000)

        for j in candidates:

            if j == i:
                continue

            D[i, j] = wasserstein_p1(x, X_train[j])

    print(f"-> Computed {np.shape(D)} distance matrix in {duration}")

    np.save(f"/Users/sela/Desktop/recsys_paper/data/screening/train/wasserstein_p1.npy", D)


def compute_distance_matrix():

    data_source = "hmm"

    for density in ["40p", "30p", "20p", "10p", "5p", "3p"]:

        X_train = np.load(f"/Users/sela/Desktop/recsys_paper/data/{data_source}/{density}/train/X_train.npy")

        # TEMP:
        #X_train = X_train[:3]

        print(f"Loaded {np.shape(X_train)} data matrix.\n")
        print("Computing distance matrix.")

        start_time = time.time()
        D = pairwise_distances(X_train, metric=wasserstein_p1)
        duration = time.time() - start_time

        print(f"Computed {np.shape(D)} distance matrix in {duration}")

        np.save(f"/Users/sela/Desktop/recsys_paper/data/{data_source}/{density}/train/wasserstein_p1.npy", D)


if __name__ == '__main__':
    approx_distance_matrix()
