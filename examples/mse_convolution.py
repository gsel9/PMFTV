"""
Simple example
"""

# local
from lmc import CMC

# third party
from sklearn.metrics import mean_squared_error
from synthetic_data import synthetic_data_generator


def main():
    rank = 5
    n_iter = 500

    for sparsity_level in [0.5, 1, 2, 3, 4]:
        for seed in [42]:
            M, X = synthetic_data_generator(
                n_rows=1000,
                n_timesteps=350,
                rank=rank,
                sparsity_level=sparsity_level,
                seed=seed,
            )

            # factorization with convolution
            mfc_model = CMC(rank=rank, n_iter=n_iter, lambda3=1)
            mfc_model.fit(X)

            # factorization without convolution
            mf_model = CMC(rank=rank, n_iter=n_iter, lambda3=0.0)
            mf_model.fit(X)

            print("With convolution:", mean_squared_error(M, mfc_model.M))
            print("Without convolution:", mean_squared_error(M, mf_model.M))


if __name__ == "__main__":
    main()
