"""
Simple example
"""

# local
from lmc import CMC
from plotting import plot_profiles_and_observations

# third party
from sklearn.metrics import mean_squared_error
from synthetic_data import synthetic_data_generator


def main():
    rank = 5

    M, X = synthetic_data_generator(n_rows=100, n_timesteps=350, rank=rank)

    plot_profiles_and_observations(X, M, path_to_fig="./figures/ground_truth_data.pdf")
    # assert afs
    # factorization with convolution
    mfc_model = CMC(rank=rank, n_iter=200)
    mfc_model.fit(X)

    plot_profiles_and_observations(
        X, mfc_model.M, path_to_fig="./figures/mfc_model.pdf"
    )

    # factorization without convolution
    mf_model = CMC(rank=rank, n_iter=200, lambda3=0.0)
    mf_model.fit(X)

    plot_profiles_and_observations(X, mf_model.M, path_to_fig="./figures/mf_model.pdf")

    print("With convolution:", mean_squared_error(M, mfc_model.M))
    print("Without convolution:", mean_squared_error(M, mf_model.M))


if __name__ == "__main__":
    main()
