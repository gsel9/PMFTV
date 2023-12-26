"""
Simple example
"""

# third party
import numpy as np

# local
# from lmc import CMC
# from plotting import plot_profiles_and_observations
from sklearn.model_selection import train_test_split
from synthetic_data import synthetic_data_generator


def regression(timepoints, X, V):
    """Inductive matrix completion via coefficient regression.

    Args:
        timepoints: Times to predict
        V: Estimated basic profiles.

    Returns:
        _type_: _description_
    """
    U_star = (2 * V.T @ X) @ np.linalg.inv(V.T @ V)
    M_hat = U_star @ V.T

    return M_hat[range(timepoints.size), timepoints]


def scm(timepoints, M):
    """Synthetic control method."""
    return


def ppd(timepoints, M):
    """Posterior predictive distribution."""
    return


def main():
    rank = 5

    M, X = synthetic_data_generator(n_rows=100, n_timesteps=250, rank=rank)

    train_idx, test_idx = train_test_split(
        range(X.shape[0]), test_Size=0.25, shuffle=False
    )

    # TODO plot predictions

    # plot_profiles_and_observations(X, M, n_profiles=4,
    # # path_to_fig="./figures/ground_truth_data.pdf")

    # factorization with convolution
    # mfc_model = CMC(rank=rank, n_iter=200)
    # mfc_model.fit(X)

    # plot_profiles_and_observations(X, mfc_model.M, n_profiles=4,
    # # path_to_fig="./figures/mfc_model.pdf")


if __name__ == "__main__":
    main()
