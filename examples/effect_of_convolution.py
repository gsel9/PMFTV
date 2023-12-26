"""
Simple example
"""

# local
from lmc import CMC, LMC
from plotting import plot_profiles_and_observations

# third party
from synthetic_data import synthetic_data_generator


def main():
    rank = 5
    n_iter = 500
    # large coefficient to emphasize regularisation effect
    lambda3 = 1

    M, X = synthetic_data_generator(
        n_rows=10, n_timesteps=350, rank=rank, sparsity_level=1
    )

    plot_profiles_and_observations(X, M, path_to_fig="./figures/ground_truth_data.pdf")

    # factorization with convolution
    mfc_model = CMC(rank=rank, n_iter=n_iter, lambda1=0.5, lambda2=0.5, lambda3=lambda3)
    mfc_model.fit(X)

    plot_profiles_and_observations(
        X, mfc_model.M, path_to_fig="./figures/mfc_model.pdf"
    )

    # factorization without convolution
    lmf_model = LMC(rank=rank, n_iter=n_iter, lambda1=0.5, lambda2=0.5, lambda3=lambda3)
    lmf_model.fit(X)

    plot_profiles_and_observations(
        X, lmf_model.M, path_to_fig="./figures/lmc_model.pdf"
    )


if __name__ == "__main__":
    main()
